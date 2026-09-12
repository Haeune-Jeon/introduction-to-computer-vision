from dataclasses import dataclass
import numpy as np

@dataclass
class BoardSpec:
    """
    rows, cols : 내부 코너 개수 (OpenCV와 동일한 정의 : (cols, rows)로  findChessboardCorners 호출)
    suqare_size : 한 칸의 한 변 길이 (단위 mm 또는 너가 정한 실제 단위)
    """
    def __init__(self, rows: int, cols: int, square_size: float):
        assert rows > 0 and cols > 0
        self.rows = rows
        self.cols = cols
        self.square_size = float(square_size)
        self.N = rows * cols
        
        # X : 가로(col 증가), Y: 세로(row 증가), Z=0
        XY = []
        for r in range(rows):
            for c in range(cols):
                X = c * self.square_size
                Y = r * self.square_size
                XY.append([X,Y])
        XY = np.asarray(XY, dtype=np.float32)  # (N,2)
        
        # 3D homogegeous M = [X, Y, Z=0,1]
        Z = np.zeros((self.N, 1), dtype=np.float32)
        ones = np.ones((self.N, 1), dtype=np.float32)
        self.M_homo = np.hstack([XY, Z, ones])   # (N,4), (X,Y,0,1)
        
        # plane homogeneous p = [X, Y, 1]
        self.p_plane = np.hstack([XY, ones])  # (N,3), (X,Y,1)
    
    def get_M_homo(self) -> np.ndarray:
        # (N,4) : [X, Y, 0, 1]
        return self.M_homo
    
    def get_p_plane(self) -> np.ndarray:
        # (N,3) : [X, Y, 1]
        return self.p_plane
