from pathlib import Path
import re
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import numpy as np

from fealpy.mesh import TriangleMesh

from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm

from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

def order_edge(mesh: TriangleMesh, start_num=0):
    edge = mesh.entity('edge')  # (NE, 2)，
    is_boundary_edge = mesh.boundary_edge_flag()
    boundary_edges = edge[is_boundary_edge]  
    
    edge_adj = {}   # 构建边的邻接表
    for u, v in boundary_edges:
        if u not in edge_adj:
            edge_adj[u] = []
        if v not in edge_adj:
            edge_adj[v] = []
        edge_adj[u].append(v)
        edge_adj[v].append(u)
    
    current = start_num   # 串联边界节点
    prev = -1
    bedge_index = [current]
    for _ in range(len(boundary_edges)):  
        next_nodes = [v for v in edge_adj[current] if v != prev]
        if not next_nodes:
            break  
        prev, current = current, next_nodes[0]
        bedge_index.append(current)
    
    if bedge_index[0] != bedge_index[-1]:  # 确保是闭合回路（首末节点一致）
        bedge_index.append(bedge_index[0])
    
    return np.array(bedge_index[:-1])  

#一.构建PDE模型
class harmonic_circle:
    @cartesian
    def source(self, p):
        """Compute exact source"""
        return 0.0


#二.读取文件数据
path = Path(r"D:\Mathematics\计算共形几何\practical_code\exercise\fealpy\data\girl.m")

if not path.exists():
    raise FileNotFoundError(f"找不到文件：{path}")

# 提取 Vertex id 和 x,y,z
vertex_pattern = re.compile(r"^Vertex\s+(\d+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
# Face 行处理
face_line_pattern = re.compile(r"^Face\s+(\d+)\s+(.+)$")
int_finder = re.compile(r"-?\d+")

vertices = {}   # id(int) -> (x,y,z)
vertex_read_order = []  
faces_raw = []  

with path.open("r", encoding="utf-8", errors="ignore") as f:
    for lineno, line in enumerate(f, start=1):
        s = line.strip()
        if not s:
            continue
        m = vertex_pattern.match(s)
        if m:
            vid = int(m.group(1))
            x = float(m.group(2)); y = float(m.group(3)); z = float(m.group(4))
            vertices[vid] = (x, y, z)
            vertex_read_order.append(vid)
            continue
        mf = face_line_pattern.match(s)
        if mf:

            nums = [int(n) for n in int_finder.findall(mf.group(2))]
            if nums:
                faces_raw.append(nums)
            else:

                print(f"[警告] 第 {lineno} 行 Face 但未找到顶点索引：{s}")

# 构建顶点数组（按 id 排序） 
sorted_ids = sorted(vertices.keys())
id_to_index = {vid: i for i, vid in enumerate(sorted_ids)}
V = np.array([vertices[vid] for vid in sorted_ids], dtype=float)  # shape (N,3)

# 将 faces_raw 转为三角形列表
triangles = []
skipped = 0
for f_ids in faces_raw:
    # 若 face 引用的 id 未在 vertices 中出现
    missing = [vid for vid in f_ids if vid not in id_to_index]
    if missing:
        print(f"[警告] Face 引用未定义顶点 id，已跳过：{missing}  原 face: {f_ids}")
        skipped += 1
        continue
    if len(f_ids) == 3:
        triangles.append((
            id_to_index[f_ids[0]],
            id_to_index[f_ids[1]],
            id_to_index[f_ids[2]],
        ))
    elif len(f_ids) > 3:
        # 组装三角形顶点数组
        v0 = id_to_index[f_ids[0]]
        for i in range(1, len(f_ids)-1):
            triangles.append((
                v0,
                id_to_index[f_ids[i]],
                id_to_index[f_ids[i+1]],
            ))
    else:
        # 少于3个顶点，需要跳过
        print(f"[警告] Face 顶点数量 < 3，已跳过：{f_ids}")
        skipped += 1

if not triangles:
    print("未找到任何三角形面。")

F = np.array(triangles, dtype=np.int64)  # shape (M,3)

#三.网格生成
mesh = TriangleMesh(V, F)
node = mesh.entity('node') #(NN, 3)

#四.建立函数空间
pde1 = harmonic_circle()
pde2 = harmonic_circle()
from fealpy.functionspace import LagrangeFESpace
p = 1
space = LagrangeFESpace(mesh, p)

#五.建立扩散积分子、双线性型并组装矩阵
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(q=3))
A0 = bform.assembly()

#六.建立源项积分子、线性型并组装矩阵
lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde1.source, q=3))
F0 = lform.assembly()

# 七.处理边界条件
oriented_bd_node_idx = order_edge(mesh,start_num=0) 

boundary_length = 0.0
for i in range(len(oriented_bd_node_idx)-1):  #计算边界总长度  
    n1 = oriented_bd_node_idx[i]
    n2 = oriented_bd_node_idx[i+1]
    boundary_length += np.linalg.norm(node[n1,:] - node[n2,:])

dirichlet1 = np.zeros(space.number_of_global_dofs())# 初始化全量数组（所有节点，边界+内部）
dirichlet2 = np.zeros(space.number_of_global_dofs())

s = 0.0
for i in range(len(oriented_bd_node_idx)-1):  # 构建边界映射
    n1 = oriented_bd_node_idx[i]
    n2 = oriented_bd_node_idx[i+1]
    segment_len = np.linalg.norm(node[n1,:] - node[n2,:])
    theta = 2.0 * np.pi * s / boundary_length
    dirichlet1[n1] = np.cos(theta)
    dirichlet2[n1] = np.sin(theta)
    s += segment_len

last_node = oriented_bd_node_idx[-1]
dirichlet1[last_node] = dirichlet1[oriented_bd_node_idx[0]]  #补充最后一个边界节点的值（闭合）
dirichlet2[last_node] = dirichlet2[oriented_bd_node_idx[0]]

bc1 = DirichletBC(space, dirichlet1)
bc2 = DirichletBC(space, dirichlet2)
A1, F1 = bc1.apply(A0, F0)
A2, F2 = bc2.apply(A0, F0)

#八.求解线性方程组
uh1 = space.function()
uh2 = space.function()
uh1[:] = spsolve(A1, F1, solver='scipy')
uh2[:] = spsolve(A2, F2, solver='scipy')
print(uh1)
print(uh2)

#九.结果可视化
circle_node = np.column_stack((uh1, uh2))  
circle_face = mesh.entity('face')  
circle_mesh = TriangleMesh(circle_node, F)  

# 对比可视化：原 3D 人脸 vs 映射后的 2D 圆盘
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')# 子图1：原 3D 人脸网格
mesh.add_plot(ax1, edgecolor='red', linewidth=0.3)
ax1.set_title('原 3D 人脸网格')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = fig.add_subplot(122)# 子图2：映射后的 2D 圆盘网格
circle_mesh.add_plot(ax2, edgecolor='blue', linewidth=0.5)
ax2.set_aspect('equal')
ax2.set_title('调和映射后的拓扑圆盘')
ax2.set_xlabel('u (cosθ)')
ax2.set_ylabel('v (sinθ)')

plt.tight_layout()
plt.show()

