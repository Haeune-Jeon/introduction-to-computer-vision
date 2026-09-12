import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path


class SfMVisualizer:
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def setup_minimal(self, figsize=(10,7), bg="white"):
        self.fig = plt.figure(figsize=figsize, facecolor=bg)
        self.ax  = self.fig.add_subplot(111, projection="3d", facecolor=bg)

        self.ax.set_axis_off()            
        self.ax.grid(False)
        
        for a in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            if hasattr(a, "set_pane_color"):
                a.set_pane_color((1, 1, 1, 0))  
            if hasattr(a, "line"):
                a.line.set_color((1, 1, 1, 0))
            a.set_ticks([])

        if hasattr(self.ax, "set_box_aspect"):
            self.ax.set_box_aspect((1, 1, 1))

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    def _equal_aspect(self):
        # 3D에서 aspect equal
        xlim = self.ax.get_xlim3d(); ylim = self.ax.get_ylim3d(); zlim = self.ax.get_zlim3d()
        xr = xlim[1]-xlim[0]; yr = ylim[1]-ylim[0]; zr = zlim[1]-zlim[0]
        mr = max(xr, yr, zr)
        xc = (xlim[0]+xlim[1])/2; yc = (ylim[0]+ylim[1])/2; zc = (zlim[0]+zlim[1])/2
        self.ax.set_xlim3d([xc-mr/2, xc+mr/2])
        self.ax.set_ylim3d([yc-mr/2, yc+mr/2])
        self.ax.set_zlim3d([zc-mr/2, zc+mr/2])
        
    def plot_sparse_cloud(self, X: np.ndarray, *, every=1, s=0.4, alpha=0.8,
                          color="#555555", view=(20, -60), clip_quantile=0.995, sample_max = 100000):
        if self.ax is None: self.setup_minimal()
        if X is None or len(X) == 0: return
        P = X[::max(1, every)]
        

        if sample_max is not None and len(P) > sample_max:
            idx = np.random.choice(len(P), sample_max, replace=False)
            P = P[idx]
            
        if clip_quantile is not None:
            q = clip_quantile
            lo = np.quantile(P, 1.0 - q, axis=0)
            hi = np.quantile(P, q, axis=0)
            pad = 0.02 * np.linalg.norm(hi - lo)
            self.ax.set_xlim(lo[0]-pad, hi[0]+pad)
            self.ax.set_ylim(lo[1]-pad, hi[1]+pad)
            self.ax.set_zlim(lo[2]-pad, hi[2]+pad)
        
        self.ax.scatter(P[:,0], P[:,1], P[:,2],
                        s=s, c=color, alpha=alpha, depthshade=False)
        self.ax.view_init(elev=view[0], azim=view[1])
        self._equal_aspect()
    
    def save_plot(self, output_path: Path):
        if self.fig is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")
    
    def show_plot(self):
        if self.fig is not None: plt.show()
    
    def clear_plot(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            