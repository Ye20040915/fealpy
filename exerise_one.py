from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
import time  # 导入时间模块
def main():
    start_time = time.time()
    ndivx, ndivy = 50, 50
    nbnd = 3
    ntotnode = ndivx * (ndivy + 2 * nbnd)
    nt = 1000
# 材料参数
    length, width = 0.05, 0.05
    holerad = 0.005
    dx = length / ndivx
    delta = 3.015 * dx
    thick = dx
    dens = 8000.0  # 因为平衡状态下pu=0不需要密度计算，此时其实没调用
    emod = 192.0e9
    area = dx * dx
    vol = area * dx
    bc = 9.0 * emod / (bm.pi * thick * (delta**3))# 键常数
    sedload1 = 9.0 / 16.0 * emod * 1.0e-6# 应变能密度1
    sedload2 = 9.0 / 16.0 * emod * 1.0e-6# 应变能密度2
    dt = 1.0
    scr0 = 0.02  
  
    # 初始化数组
    coord = bm.zeros((ntotnode, 2))
    numfam = bm.zeros(ntotnode, dtype=int)
    pointfam = bm.zeros(ntotnode, dtype=int)
    pforce = bm.zeros((ntotnode, 2))# 近场力
    pforceold = bm.zeros((ntotnode, 2))# 旧近场力
    stendens = bm.zeros((ntotnode, 2))# 体积力
    fncst = bm.ones((ntotnode, 2))
    disp = bm.zeros((ntotnode, 2))
    vel = bm.zeros((ntotnode, 2))
    velhalfold = bm.zeros((ntotnode, 2)) # 半步速度
    velhalf = bm.zeros((ntotnode, 2))  # 新半步速度
    massvec = bm.zeros((ntotnode, 2))  # 质量向量
    dmg = bm.zeros(ntotnode) # 损伤
    fail = bm.ones((ntotnode, 100), dtype=int) # 破坏标志
    
    nodefam = []
  
    # 设置材料点坐标
    nnum = 0
    for i in range(ndivy):
        for j in range(ndivx):
            coordx = -length/2 + dx/2 + j*dx
            coordy = -width/2 + dx/2 + i*dx
            tmprad = bm.sqrt(coordx**2 + coordy**2)
            if tmprad > holerad:
                coord[nnum] = [coordx, coordy]
                nnum += 1
    totint = nnum
    
    # 边界区域
    for i in range(nbnd):
        for j in range(ndivx):
            coord[nnum] = [-length/2 + dx/2 + j*dx, -width/2 - dx/2 - i*dx]
            nnum += 1
    totbottom = nnum
    
    for i in range(nbnd):
        for j in range(ndivx):
            coord[nnum] = [-length/2 + dx/2 + j*dx, width/2 + dx/2 + i*dx]
            nnum += 1
    tottop = nnum
    totnode = nnum
    # 确定家族成员
    pointfam[0] = 0
    for i in range(totnode):
        fam_members = []
        for j in range(totnode):
            if i != j:
                dist = bm.linalg.norm(coord[j] - coord[i])
                if dist <= delta:
                    fam_members.append(j)
        numfam[i] = len(fam_members)
        if i > 0:
            pointfam[i] = pointfam[i-1] + numfam[i-1]
        nodefam.extend(fam_members)
    
    radij = dx / 2

    # 计算表面修正因子
    for i in range(totnode):
        disp[i] = [0.001 * coord[i,0], 0.0]
    
    for i in range(totnode):
        stendens[i,0] = 0.0
        for j in range(numfam[i]):
            cnode = nodefam[pointfam[i] + j]
            idist = bm.linalg.norm(coord[cnode] - coord[i])
            nlength = bm.linalg.norm(coord[cnode] + disp[cnode] - coord[i] - disp[i])
            
            if idist <= delta - radij:
                fac = 1.0
            elif idist <= delta + radij:
                fac = (delta + radij - idist) / (2.0 * radij)
            else:
                fac = 0.0
                
            stendens[i,0] += 0.5 * 0.5 * bc * ((nlength - idist) / idist)**2 * idist * vol * fac
            
        fncst[i,0] = sedload1 / stendens[i,0] if stendens[i,0] != 0 else 1.0
    
    for i in range(totnode):
        disp[i] = [0.0, 0.001 * coord[i,1]]
    
    for i in range(totnode):
        stendens[i,1] = 0.0
        for j in range(numfam[i]):
            cnode = nodefam[pointfam[i] + j]
            idist = bm.linalg.norm(coord[cnode] - coord[i])
            nlength = bm.linalg.norm(coord[cnode] + disp[cnode] - coord[i] - disp[i])
            
            if idist <= delta - radij:
                fac = 1.0
            elif idist <= delta + radij:
                fac = (delta + radij - idist) / (2.0 * radij)
            else:
                fac = 0.0
                
            stendens[i,1] += 0.5 * 0.5 * bc * ((nlength - idist) / idist)**2 * idist * vol * fac
            
        fncst[i,1] = sedload2 / stendens[i,1] if stendens[i,1] != 0 else 1.0
    
    # 初始化位移和速度
    disp.fill(0.0)
    vel.fill(0.0)
    
    # 质量向量
    mass_val = 0.25 * dt * dt * (bm.pi * delta**2 * thick) * bc / dx
    massvec[:,0] = mass_val
    massvec[:,1] = mass_val
    
    # 存储四个关键时间步的结果
    results = {}
    # 时间积分
    for tt in range(1, nt+1):
        if tt % 100 == 0:
            print(f'时间步: {tt}/{nt}')
        
        # 边界条件
        for i in range(totint, totbottom):
            vel[i,1] = -2.7541e-7
            disp[i,1] = -2.7541e-7 * tt * dt
            
        for i in range(totbottom, tottop):
            vel[i,1] = 2.7541e-7
            disp[i,1] = 2.7541e-7 * tt * dt
        
        # 计算近场力
        for i in range(totint):
            dmgpar1, dmgpar2 = 0.0, 0.0
            pforce[i] = [0.0, 0.0]
            
            for j in range(numfam[i]):
                cnode = nodefam[pointfam[i] + j]
                idist = bm.linalg.norm(coord[cnode] - coord[i])
                nlength = bm.linalg.norm(coord[cnode] + disp[cnode] - coord[i] - disp[i])
                
                if idist <= delta - radij:
                    fac = 1.0
                elif idist <= delta + radij:
                    fac = (delta + radij - idist) / (2.0 * radij)
                else:
                    fac = 0.0
                
                dx_val = coord[cnode,0] - coord[i,0]
                dy_val = coord[cnode,1] - coord[i,1]
                if abs(dy_val) <= 1e-10:
                    theta = 0.0
                elif abs(dx_val) <= 1e-10:
                    theta = bm.pi / 2
                else:
                    theta = bm.atan(abs(dy_val) / abs(dx_val))
                
                scx = (fncst[i,0] + fncst[cnode,0]) / 2.0
                scy = (fncst[i,1] + fncst[cnode,1]) / 2.0
                scr = 1.0 / ((bm.cos(theta)**2 / scx**2) + (bm.sin(theta)**2 / scy**2))
                scr = bm.sqrt(scr)
                
                if fail[i,j] == 1:
                    dir_vec = (coord[cnode] + disp[cnode] - coord[i] - disp[i]) / nlength
                    force_mag = bc * (nlength - idist) / idist * vol * scr * fac
                    dforce = force_mag * dir_vec
                else:
                    dforce = [0.0, 0.0]
                    
                pforce[i] += dforce
                
                stretch = abs(nlength - idist) / idist
                if stretch > scr0 and abs(coord[i,1]) <= length/4.0:
                    fail[i,j] = 0
                
                dmgpar1 += fail[i,j] * vol * fac
                dmgpar2 += vol * fac
            
            dmg[i] = 1.0 - dmgpar1 / dmgpar2 if dmgpar2 != 0 else 0.0
        
        # 自适应动态松弛
        cn1, cn2 = 0.0, 0.0
        for i in range(totint):
            if velhalfold[i,0] != 0.0:
                cn1 -= disp[i,0]**2 * (pforce[i,0]/massvec[i,0] - pforceold[i,0]/massvec[i,0]) / (dt * velhalfold[i,0])
            if velhalfold[i,1] != 0.0:
                cn1 -= disp[i,1]**2 * (pforce[i,1]/massvec[i,1] - pforceold[i,1]/massvec[i,1]) / (dt * velhalfold[i,1])
            
            cn2 += disp[i,0]**2 + disp[i,1]**2
        
        cn = 2.0 * bm.sqrt(cn1/cn2) if cn2 != 0 and cn1/cn2 > 0 else 0.0
        cn = min(cn, 1.9)
        
        # 更新运动状态
        for i in range(totint):
            if tt == 1:
                velhalf[i] = dt / massvec[i] * (pforce[i]) / 2.0
            else:
                denom = 2.0 + cn * dt
                velhalf[i] = ((2.0 - cn * dt) * velhalfold[i] + 2.0 * dt / massvec[i] * pforce[i]) / denom
            
            vel[i] = 0.5 * (velhalfold[i] + velhalf[i])
            disp[i] += velhalf[i] * dt
            
            velhalfold[i] = velhalf[i]
            pforceold[i] = pforce[i]
        
        # 保存四个关键时间步的结果
        if tt in [675, 750, 850, 1000]:  
            results[tt] = {
                'disp': disp[:totint].copy(),
                'dmg': dmg[:totint].copy(),
                'fail': fail[:totint].copy()
            } 
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总计算时间: {total_time:.2f} 秒")       
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (tt, result) in enumerate(results.items()):
        ax = axes[idx]
        sc = ax.scatter(coord[:totint, 0], coord[:totint, 1], c=result['dmg'], 
                       cmap='jet', s=20, vmin=0, vmax=1)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title(f'时间步 {tt} 的损伤分布')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='损伤值')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()