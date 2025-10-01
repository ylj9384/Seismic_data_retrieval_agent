# dynamic tool: calculate_pga_kanai_1966
def calculate_pga_kanai_1966(magnitude: float, distance: float, period: float) -> dict:
    """Kanai (1966) PGA 模型计算函数
    参数:
        magnitude: 震级 (M)
        distance: 震中距 (km)
        period: 场地固有周期 (秒)
    返回:
        包含 PGA 值及 P、Q 解释的字典
    """
    import math
    try:
        # Kanai (1966) 模型简化形式（典型参数）：PGA = P * 10^(Q*M) / (distance + 1)
        # P 和 Q 为经验常数，P 反映区域衰减和场地放大效应，Q 控制震级对 PGA 的指数增长
        P = 0.25  # 典型值，单位 cm/s²，受局部地质和传播路径影响
        Q = 0.5   # 典型值，无量纲，控制震级敏感度

        if distance < 0 or magnitude < 0 or period <= 0:
            return {'status': 'error', 'reason': '输入参数必须为正数'}

        # 根据固有周期调整放大系数（近似：共振增强 PGA）
        # 假设在 T=0.5s 附近有最大响应，使用高斯型增益
        T_res = 0.5  # 共振周期假设（秒）
        sigma_T = 0.2
        amplification = math.exp(-((period - T_res)**2) / (2 * sigma_T**2)) + 1.0  # 基础增益

        # 计算基本 PGA
        pga_cms2 = (P * (10 ** (Q * magnitude))) / (distance + 1) * amplification
        pga_g = pga_cms2 / 980.0  # 转换为 g 单位

        return {
            'status': 'success',
            'pga_cms2': round(pga_cms2, 3),
            'pga_g': round(pga_g, 5),
            'parameters': {
                'P': P,
                'Q': Q,
                'explanation': {
                    'P_meaning': 'P 是与区域地震传播特性、衰减和局部场地条件相关的比例系数。P 值越大，表示区域整体 PGA 水平越高，可能反映松软沉积层或低阻尼特征。',
                    'Q_meaning': 'Q 是震级敏感系数，控制 PGA 随震级指数增长的速度。Q 值越大，大震时 PGA 增长越快。',
                    'impact_on_result': '本例中 P=0.25 和 Q=0.5 为典型参考值；若实际地区 P 更高（如软土），PGA 将显著增大；Q 若偏高，则 M6.0 的能量释放对 PGA 贡献更强。'
                }
            },
            'input': {
                'magnitude': magnitude,
                'distance_km': distance,
                'site_period_s': period
            }
        }
    except Exception as e:
        return {'status': 'error', 'reason': f'计算失败: {str(e)}'}
