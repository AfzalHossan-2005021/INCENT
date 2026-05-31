"""
synthesize.py
===================
Interactive *rotatable* bounding-box crop for MERFISH / spatial-transcriptomics
AnnData slices, with built-in AnnData (.h5ad) save support.

Mirrors the AnnData contract and ``.uns["self_alignment_test"]`` layout of
``core.create_random_rectangular_portion`` — including the ``window_angle_radians``
field — so the result slots directly into ``pairwise_align`` /
``hierarchical_pairwise_align`` without any adaptation.

Rotation math is identical to ``create_random_rectangular_portion``:
    local = (coords - center) @ R,   R = [[cos θ, -sin θ], [sin θ, cos θ]]
    mask  = |local[:,0]| ≤ half_width  AND  |local[:,1]| ≤ half_height

Workflow (Jupyter notebook)
---------------------------
    %matplotlib widget          # ipympl — pip install ipympl if needed
    from synthesize import create_interactive_rectangular_portion

    selector = create_interactive_rectangular_portion(adata)
    # 1. Drag a rectangle over the hemisphere you want
    # 2. Drag the Rotation slider to orient the box
    # 3. Type a file path in the Save-path box and click 💾 Save AnnData
    # 4. Click ✓ Confirm selection

    # In the next cell:
    portion = selector.extract()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import RectangleSelector, Button, Slider, TextBox
from anndata import AnnData


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_coords(adata: AnnData, spatial_key: str) -> np.ndarray:
    """Return validated (N, 2) float64 spatial coordinates."""
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"spatial_key '{spatial_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be (N, 2+); got {coords.shape}."
        )
    return coords[:, :2]


def _make_color_array(
    adata: AnnData,
    label_key: Optional[str],
) -> Tuple[np.ndarray, Optional[dict]]:
    """Per-cell RGBA colour array + legend dict keyed by label."""
    if label_key is not None and label_key in adata.obs.columns:
        labels  = adata.obs[label_key].astype(str).values
        unique  = sorted(set(labels))
        cmap    = plt.get_cmap("tab20", max(len(unique), 1))
        lbl2idx = {l: i for i, l in enumerate(unique)}
        colors  = np.array([cmap(lbl2idx[l]) for l in labels])
        legend  = {l: cmap(lbl2idx[l]) for l in unique}
        return colors, legend
    colors = np.full((adata.n_obs, 4), [0.5, 0.5, 0.5, 0.6])
    return colors, None


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    """2×2 counter-clockwise rotation matrix — identical to core.py convention."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rotated_polygon_corners(
    cx: float, cy: float,
    hw: float, hh: float,
    angle_rad: float,
) -> np.ndarray:
    """
    Return (4, 2) world-frame corners of the rotated rectangle.
    Local frame: ±hw along local-x, ±hh along local-y.
    """
    R = _rotation_matrix(angle_rad)
    local_corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float64)
    # R maps local → world: world = local @ R.T  (same as core.py's `local = centered @ R`)
    return local_corners @ R.T + np.array([cx, cy])


def _compute_rotated_mask(
    coords: np.ndarray,
    cx: float, cy: float,
    hw: float, hh: float,
    angle_rad: float,
) -> np.ndarray:
    """
    Boolean mask of cells inside the rotated rectangle.

    Uses the same transform as ``create_random_rectangular_portion``:
        local = (coords - center) @ R
    so results are numerically identical.
    """
    R       = _rotation_matrix(angle_rad)
    centered = coords - np.array([cx, cy])
    local   = centered @ R           # project into box frame
    return (np.abs(local[:, 0]) <= hw) & (np.abs(local[:, 1]) <= hh)


def _build_metadata(
    cx: float, cy: float,
    hw: float, hh: float,
    angle_rad: float,
    portion: AnnData,
    source_n_obs: int,
    save_path: Optional[str],
) -> dict:
    """
    ``.uns["self_alignment_test"]`` dict — fully compatible with
    ``create_random_rectangular_portion`` including ``window_angle_radians``.
    """
    pct = 100.0 * portion.n_obs / max(source_n_obs, 1)
    return {
        # ── keys shared with create_random_rectangular_portion ──
        "random_seed":          None,          # user-defined crop has no seed
        "portion_percentage":   float(pct),
        "selected_cell_count":  int(portion.n_obs),
        "window_center":        [float(cx), float(cy)],
        "window_angle_radians": float(angle_rad),
        "window_width":         float(2 * hw),
        "window_height":        float(2 * hh),
        "window_aspect_ratio":  float(hw / max(hh, 1e-12)),
        # ── extra provenance for interactive crops ──
        "crop_mode":            "interactive_rotatable",
        "window_angle_degrees": float(np.degrees(angle_rad)),
        "window_half_width":    float(hw),
        "window_half_height":   float(hh),
        "save_path":            save_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveCropSelector:
    """
    Interactive MERFISH spatial crop selector with rotation and save support.

    Parameters
    ----------
    adata : AnnData
        Full slice.
    spatial_key : str
        Key in ``adata.obsm`` for (N, 2+) coordinates.
    label_key : str or None
        ``adata.obs`` column for scatter colour; ``None`` → monochrome.
    point_size : float
        Scatter point size (matplotlib ``s``).
    alpha : float
        Scatter point transparency.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str or None
        Figure title; auto-generated if ``None``.
    min_cells : int
        Minimum cells the crop must contain.
    default_save_path : str
        Pre-filled value in the Save-path text box.
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        label_key: Optional[str] = "cell_type_annot",
        point_size: float = 2.0,
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (11, 10),
        title: Optional[str] = None,
        min_cells: int = 4,
        default_save_path: str = "portion.h5ad",
    ):
        self.adata            = adata
        self.spatial_key      = spatial_key
        self.label_key        = label_key
        self.min_cells        = min_cells

        self._coords          = _get_coords(adata, spatial_key)       # (N,2) float64
        self._colors, self._legend = _make_color_array(adata, label_key)

        # ── selection state ──────────────────────────────────────────────────
        # After RectangleSelector: canonical box in local (unrotated) frame.
        self._cx: Optional[float] = None   # centre x
        self._cy: Optional[float] = None   # centre y
        self._hw: Optional[float] = None   # half-width  (local x extent)
        self._hh: Optional[float] = None   # half-height (local y extent)
        self._angle_rad: float    = 0.0    # current rotation

        self._poly_patch: Optional[MplPolygon] = None   # rotated rectangle overlay
        self._portion: Optional[AnnData]        = None
        self._confirmed: bool                   = False
        self._last_save_path: Optional[str]     = None

        # ── figure layout ────────────────────────────────────────────────────
        # Reserve bottom space:
        #   row 0 (top-most widget row): rotation slider   ~0.265 → 0.295
        #   row 1: save-path textbox                       ~0.165 → 0.205
        #   row 2: buttons                                 ~0.040 → 0.110
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.subplots_adjust(bottom=0.34, top=0.94)

        # ── scatter plot ──────────────────────────────────────────────────────
        self._scatter = self.ax.scatter(
            self._coords[:, 0], self._coords[:, 1],
            c=self._colors, s=point_size, alpha=alpha,
            linewidths=0, rasterized=True,
        )
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("X coordinate (µm)", fontsize=9)
        self.ax.set_ylabel("Y coordinate (µm)", fontsize=9)
        self.ax.set_title(
            title or
            f"MERFISH slice  ·  {adata.n_obs:,} cells  —  "
            "① Drag rectangle   ② Rotate slider   ③ Save / Confirm",
            fontsize=10,
        )

        # cell-type legend
        if self._legend:
            handles = [
                mpatches.Patch(color=c, label=lbl)
                for lbl, c in list(self._legend.items())[:20]
            ]
            if len(self._legend) > 20:
                handles.append(mpatches.Patch(
                    color="none", label=f"… +{len(self._legend)-20} more"))
            self.ax.legend(
                handles=handles, loc="upper right",
                fontsize=6, markerscale=2, framealpha=0.6,
                title=label_key, title_fontsize=7,
            )

        # status bar (inside axes, top-left)
        self._status_text = self.ax.text(
            0.01, 0.99,
            "🖱  Click-and-drag to draw the initial rectangle",
            transform=self.ax.transAxes,
            va="top", ha="left", fontsize=8.5, color="navy",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="lightyellow", ec="steelblue", alpha=0.88),
        )

        # ── RectangleSelector ────────────────────────────────────────────────
        self._rs = RectangleSelector(
            self.ax, self._on_select,
            useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="pixels",
            interactive=True,
            props=dict(facecolor="steelblue", edgecolor="navy",
                       alpha=0.18, linewidth=1.8, linestyle="--"),
        )

        # ── Rotation Slider ──────────────────────────────────────────────────
        # [left, bottom, width, height] in figure-fraction coordinates
        ax_slider = self.fig.add_axes([0.12, 0.255, 0.76, 0.030])
        self._slider = Slider(
            ax_slider, "Rotation (°)",
            valmin=0.0, valmax=180.0, valinit=0.0,
            valfmt="%.1f°",
            color="steelblue",
        )
        self._slider.on_changed(self._on_slider_change)

        # ── Save-path TextBox ────────────────────────────────────────────────
        # Label sits to the left of the textbox
        self.fig.text(
            0.12, 0.195, "Save path (.h5ad):",
            ha="left", va="center", fontsize=8.5,
        )
        ax_textbox = self.fig.add_axes([0.35, 0.175, 0.53, 0.038])
        self._textbox = TextBox(
            ax_textbox, "",
            initial=default_save_path,
            textalignment="left",
        )
        # TextBox does not fire on every keystroke — value is read on Save click

        # ── Buttons ──────────────────────────────────────────────────────────
        #  [Reset]   [💾 Save AnnData]   [✓ Confirm]
        ax_btn_reset   = self.fig.add_axes([0.05, 0.055, 0.17, 0.075])
        ax_btn_save    = self.fig.add_axes([0.33, 0.055, 0.28, 0.075])
        ax_btn_confirm = self.fig.add_axes([0.68, 0.055, 0.27, 0.075])

        self._btn_reset   = Button(ax_btn_reset,   "↺  Reset",
                                   color="0.86", hovercolor="0.75")
        self._btn_save    = Button(ax_btn_save,    "💾  Save AnnData (.h5ad)",
                                   color="#fff3cd", hovercolor="#ffe08a")
        self._btn_confirm = Button(ax_btn_confirm, "✓  Confirm selection",
                                   color="#d4edda", hovercolor="#9dd6aa")

        self._btn_reset.on_clicked(self._on_reset)
        self._btn_save.on_clicked(self._on_save)
        self._btn_confirm.on_clicked(self._on_confirm)

        plt.show()
        self._print_instructions()

    # ─────────────────────────────────────────────────────────────────────────
    # Widget callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_select(self, eclick, erelease):
        """RectangleSelector callback — stores canonical box, refreshes overlay."""
        x0, y0 = eclick.xdata,   eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        self._cx = (x_min + x_max) / 2.0
        self._cy = (y_min + y_max) / 2.0
        self._hw = (x_max - x_min) / 2.0
        self._hh = (y_max - y_min) / 2.0

        # Reset rotation to 0 whenever a new drag is started
        self._slider.set_val(0.0)   # triggers _on_slider_change → _refresh

    def _on_slider_change(self, val: float):
        """Rotation-slider callback — updates angle and redraws."""
        self._angle_rad = float(np.radians(val))
        self._refresh()

    def _on_reset(self, event):
        """Clear selection entirely."""
        self._cx = self._cy = self._hw = self._hh = None
        self._angle_rad  = 0.0
        self._portion    = None
        self._confirmed  = False

        self._slider.set_val(0.0)
        self._remove_poly_patch()
        self._rs.set_active(True)

        self._status_text.set_text(
            "🖱  Selection reset — drag a new rectangle")
        self._status_text.get_bbox_patch().set_facecolor("lightyellow")
        self.fig.canvas.draw_idle()

    def _on_save(self, event):
        """Save the current (pre-confirm) crop to disk as .h5ad."""
        if not self._has_box():
            print("⚠  Draw a rectangle first before saving.")
            return
        path = self._textbox.text.strip()
        if not path:
            print("⚠  Enter a valid file path in the Save-path box.")
            return
        if not path.endswith(".h5ad"):
            path += ".h5ad"
        try:
            portion = self._crop()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            portion.write_h5ad(path)
            self._last_save_path = path
            n_cells = portion.n_obs
            angle_deg = np.degrees(self._angle_rad)
            self._status_text.set_text(
                f"💾  Saved  {n_cells:,} cells  ·  θ={angle_deg:.1f}°  →  {path}")
            self._status_text.get_bbox_patch().set_facecolor("#cce5ff")
            print(f"\n💾  Saved {n_cells:,} cells → '{path}'\n")
        except Exception as err:
            self._status_text.set_text(f"❌  Save failed: {err}")
            self._status_text.get_bbox_patch().set_facecolor("#f8d7da")
            print(f"❌  Save error: {err}")
        self.fig.canvas.draw_idle()

    def _on_confirm(self, event):
        """Confirm selection; portion available via .extract()."""
        if not self._has_box():
            print("⚠  Draw a rectangle first.")
            return
        try:
            self._portion   = self._crop()
            self._confirmed = True
            n    = self._portion.n_obs
            pct  = 100.0 * n / self.adata.n_obs
            deg  = np.degrees(self._angle_rad)
            self._status_text.set_text(
                f"✓  Confirmed  —  {n:,} cells ({pct:.1f}%)  ·  θ={deg:.1f}°  "
                f"→  call .extract()")
            self._status_text.get_bbox_patch().set_facecolor("#d4edda")
            self.fig.canvas.draw_idle()
            print(
                f"\n✅  Crop confirmed:\n"
                f"   Cells     : {n:,} / {self.adata.n_obs:,}  ({pct:.1f}%)\n"
                f"   Angle     : {deg:.2f}°\n"
                f"   Centre    : ({self._cx:.1f}, {self._cy:.1f})\n"
                f"   W × H     : {2*self._hw:.1f} × {2*self._hh:.1f} µm\n"
                f"\n   Run  `portion = selector.extract()`  in the next cell.\n"
            )
        except ValueError as err:
            print(f"⚠  {err}")

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh(self):
        """Recompute mask, update polygon overlay and status bar."""
        if not self._has_box():
            return

        # ── rotated polygon overlay ──────────────────────────────────────────
        corners = _rotated_polygon_corners(
            self._cx, self._cy, self._hw, self._hh, self._angle_rad)
        self._remove_poly_patch()
        self._poly_patch = MplPolygon(
            corners, closed=True,
            facecolor="steelblue", edgecolor="navy",
            alpha=0.28, linewidth=2.0, zorder=4,
        )
        self.ax.add_patch(self._poly_patch)

        # ── live cell count ──────────────────────────────────────────────────
        mask  = _compute_rotated_mask(
            self._coords,
            self._cx, self._cy, self._hw, self._hh, self._angle_rad,
        )
        n_in  = int(mask.sum())
        pct   = 100.0 * n_in / self.adata.n_obs
        deg   = np.degrees(self._angle_rad)
        ok    = n_in >= self.min_cells

        self._status_text.set_text(
            f"θ = {deg:.1f}°  ·  {n_in:,} cells  ({pct:.1f}%)"
            + ("" if ok else f"  ⚠ need ≥ {self.min_cells}")
        )
        self._status_text.get_bbox_patch().set_facecolor(
            "#d4edda" if ok else "#f8d7da")

        self.fig.canvas.draw_idle()

    def _remove_poly_patch(self):
        if self._poly_patch is not None:
            try:
                self._poly_patch.remove()
            except ValueError:
                pass
            self._poly_patch = None

    def _has_box(self) -> bool:
        return None not in (self._cx, self._cy, self._hw, self._hh)

    # ─────────────────────────────────────────────────────────────────────────
    # Crop computation
    # ─────────────────────────────────────────────────────────────────────────

    def _crop(self) -> AnnData:
        """Internal: apply current box + angle to adata, return cropped copy."""
        mask    = _compute_rotated_mask(
            self._coords,
            self._cx, self._cy, self._hw, self._hh, self._angle_rad,
        )
        indices = np.flatnonzero(mask)
        n_in    = len(indices)

        if n_in < self.min_cells:
            raise ValueError(
                f"Only {n_in} cell(s) inside the box "
                f"(minimum: {self.min_cells}).  Draw a larger rectangle."
            )
        if n_in >= self.adata.n_obs:
            warnings.warn(
                "The selection covers the entire slice.  "
                "Consider a smaller or better-oriented box.",
                UserWarning, stacklevel=3,
            )

        portion = self.adata[indices].copy()
        portion.uns["self_alignment_test"] = _build_metadata(
            self._cx, self._cy, self._hw, self._hh,
            self._angle_rad, portion, self.adata.n_obs,
            self._last_save_path,
        )
        return portion

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self) -> AnnData:
        """
        Return the confirmed cropped AnnData.

        Raises ``RuntimeError`` if called before the ✓ Confirm button is clicked.
        """
        if not self._confirmed or self._portion is None:
            raise RuntimeError(
                "No confirmed selection yet.  "
                "Draw a rectangle and click '✓ Confirm selection' first."
            )
        return self._portion

    def get_state(self) -> dict:
        """Return the current crop parameters (useful for logging / reproducibility)."""
        return {
            "cx": self._cx,
            "cy": self._cy,
            "half_width":  self._hw,
            "half_height": self._hh,
            "angle_rad":   self._angle_rad,
            "angle_deg":   float(np.degrees(self._angle_rad)),
            "confirmed":   self._confirmed,
            "save_path":   self._last_save_path,
        }

    def crop_from_params(
        self,
        cx: float, cy: float,
        half_width: float, half_height: float,
        angle_rad: float = 0.0,
    ) -> AnnData:
        """
        Programmatic crop — replays a previously recorded selection without GUI.

        Parameters match ``adata.uns["self_alignment_test"]``:
            cx, cy          ← ``window_center``
            half_width      ← ``window_half_width``
            half_height     ← ``window_half_height``
            angle_rad       ← ``window_angle_radians``

        Returns
        -------
        AnnData
            Cropped slice with full ``.uns`` metadata.
        """
        self._cx, self._cy = float(cx), float(cy)
        self._hw, self._hh = float(half_width), float(half_height)
        self._angle_rad    = float(angle_rad)
        return self._crop()

    @staticmethod
    def _print_instructions():
        print(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Interactive MERFISH Crop Selector — Instructions\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ①  Click-and-drag on the scatter to draw a rectangle.\n"
            "      Drag handles let you resize after release.\n"
            "  ②  Use the Rotation slider to orient the box (0–180°).\n"
            "      The blue polygon and cell count update in real time.\n"
            "  ③  (Optional) Edit the Save-path box, then click\n"
            "      💾 Save AnnData to write the crop to disk as .h5ad.\n"
            "  ④  Click  ✓ Confirm selection  when satisfied.\n"
            "  ⑤  In the next notebook cell, run:\n"
            "          portion = selector.extract()\n"
            "  ↺  Reset clears everything so you can start over.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def create_interactive_rectangular_portion(
    adata: AnnData,
    spatial_key: str = "spatial",
    label_key: Optional[str] = "cell_type_annot",
    point_size: float = 2.0,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (11, 10),
    title: Optional[str] = None,
    min_cells: int = 4,
    default_save_path: str = "portion.h5ad",
) -> InteractiveCropSelector:
    """
    Open the interactive rotatable crop selector and return the selector object.

    This is the interactive, rotatable analogue of
    ``core.create_random_rectangular_portion``.  Non-blocking: call
    ``.extract()`` in a subsequent notebook cell after confirming.

    Parameters
    ----------
    adata : AnnData
        Full MERFISH slice.
    spatial_key : str
        Key in ``adata.obsm`` for (N, 2+) spatial coordinates.
    label_key : str or None
        ``adata.obs`` column for scatter colour coding.
    point_size : float
        Scatter point size.
    alpha : float
        Scatter transparency (0–1).
    figsize : tuple of int
        Figure (width, height) in inches.
    title : str or None
        Custom figure title.
    min_cells : int
        Minimum cells the crop must contain.
    default_save_path : str
        Pre-filled value in the Save-path text box.

    Returns
    -------
    InteractiveCropSelector

    Examples
    --------
    ::

        %matplotlib widget
        from synthesize import create_interactive_rectangular_portion

        selector = create_interactive_rectangular_portion(
            adata,
            label_key="cell_type_annot",
            default_save_path="results/left_hemisphere.h5ad",
        )
        # — draw, rotate, save, confirm —

        # next cell:
        portion = selector.extract()
        # → pass to hierarchical_pairwise_align(sliceA=portion, ...)
    """
    return InteractiveCropSelector(
        adata=adata,
        spatial_key=spatial_key,
        label_key=label_key,
        point_size=point_size,
        alpha=alpha,
        figsize=figsize,
        title=title,
        min_cells=min_cells,
        default_save_path=default_save_path,
    )


def select_rectangular_portion_blocking(
    adata: AnnData,
    spatial_key: str = "spatial",
    label_key: Optional[str] = "cell_type_annot",
    **kwargs,
) -> AnnData:
    """
    Blocking version for plain Python scripts.

    Opens the plot, polls until the user clicks ✓ Confirm, then returns
    the cropped AnnData.  In Jupyter, prefer ``create_interactive_rectangular_portion``.
    """
    selector = InteractiveCropSelector(
        adata=adata, spatial_key=spatial_key, label_key=label_key, **kwargs)
    while not selector._confirmed and plt.fignum_exists(selector.fig.number):
        plt.pause(0.1)
    if not selector._confirmed:
        raise RuntimeError(
            "Figure closed before confirmation.  Re-run and click ✓ Confirm.")
    return selector.extract()


# ─────────────────────────────────────────────────────────────────────────────
# Preview helper (rotation-aware)
# ─────────────────────────────────────────────────────────────────────────────

def preview_crop(
    adata_full: AnnData,
    portion: AnnData,
    spatial_key: str = "spatial",
    label_key: Optional[str] = "cell_type_annot",
    figsize: Tuple[int, int] = (13, 6),
) -> None:
    """
    Side-by-side view of the full slice (with the rotated crop polygon) and
    the extracted portion.  Reads rotation from ``portion.uns``.

    Parameters
    ----------
    adata_full : AnnData
        Original full slice.
    portion : AnnData
        Cropped AnnData returned by ``selector.extract()``.
    """
    coords_full,    _           = _get_coords(adata_full, spatial_key), None
    colors_full,    _           = _make_color_array(adata_full, label_key)
    coords_full                 = _get_coords(adata_full, spatial_key)
    coords_crop                 = _get_coords(portion, spatial_key)
    colors_crop,    legend      = _make_color_array(portion, label_key)

    meta  = portion.uns.get("self_alignment_test", {})
    cx    = meta.get("window_center",        [coords_crop[:,0].mean()])[0]
    cy    = meta.get("window_center",        [None, coords_crop[:,1].mean()])[1]
    hw    = meta.get("window_half_width",    np.ptp(coords_crop[:,0]) / 2)
    hh    = meta.get("window_half_height",   np.ptp(coords_crop[:,1]) / 2)
    angle = meta.get("window_angle_radians", 0.0)
    pct   = meta.get("portion_percentage",
                     100.0 * portion.n_obs / adata_full.n_obs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── left: full slice + rotated crop box ──────────────────────────────────
    ax1.scatter(
        coords_full[:, 0], coords_full[:, 1],
        c=colors_full, s=1.5, alpha=0.4,
        linewidths=0, rasterized=True,
    )
    if None not in (cx, cy, hw, hh):
        corners = _rotated_polygon_corners(cx, cy, hw, hh, angle)
        poly = MplPolygon(
            corners, closed=True,
            facecolor="none", edgecolor="crimson",
            linewidth=2, linestyle="--", zorder=5,
        )
        ax1.add_patch(poly)
        ax1.plot(*np.array([cx, cy]), "r+", ms=10, mew=2, zorder=6)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title(
        f"Full slice  ({adata_full.n_obs:,} cells)\n"
        f"Dashed polygon = crop  ·  θ = {np.degrees(angle):.1f}°",
        fontsize=10,
    )
    ax1.set_xlabel("X (µm)")
    ax1.set_ylabel("Y (µm)")

    # ── right: extracted crop ─────────────────────────────────────────────────
    ax2.scatter(
        coords_crop[:, 0], coords_crop[:, 1],
        c=colors_crop, s=3, alpha=0.8,
        linewidths=0, rasterized=True,
    )
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title(
        f"Extracted crop  ({portion.n_obs:,} cells  ·  {pct:.1f}%  ·  "
        f"θ = {np.degrees(angle):.1f}°)",
        fontsize=10,
    )
    ax2.set_xlabel("X (µm)")
    ax2.set_ylabel("Y (µm)")

    if legend:
        handles = [mpatches.Patch(color=c, label=l)
                   for l, c in list(legend.items())[:18]]
        ax2.legend(handles=handles, fontsize=6, loc="upper right",
                   framealpha=0.6, title=label_key, title_fontsize=7)

    plt.suptitle("Interactive Rotatable Crop — Preview", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_dir = "../data/"
    data_file = "adata90wk_donor_id_3_slice_0.h5ad"

    adata = sc.read_h5ad(data_dir + data_file)

    selector = create_interactive_rectangular_portion(
        adata,                       # your full MERFISH AnnData
        spatial_key="spatial",
        label_key="cell_type_annot", # colours scatter by cell type
        point_size=2,
        figsize=(11, 9),
        default_save_path=data_dir + "synthetic/" + data_file.replace(".h5ad", "_portion.h5ad")
    )
    
    portion = selector.extract()
    print(portion)  # AnnData with N_obs = cells in bbox
    preview_crop(adata, portion, label_key="cell_type_annot")
