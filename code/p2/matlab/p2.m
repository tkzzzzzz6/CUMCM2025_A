%% methodA_optimize_problem2.m
% 题目A - 问题2：方法A（粗到细 + 无导数局部微调）求解单弹遮蔽时间最大化
% 变量：theta（航向角偏移）、v（速度）、t_r（投放时刻）、tau_f（引信延迟）
% 目标：最大化起爆后20s内的“有效遮蔽”总时长（云团中心到 M(t)->T 视线段距离<=10m 且落在线段上）
% 可视化：最优方案的XY俯视图 + 距离曲线（含遮蔽区间）

clear; clc; close all;

%% ===================== 常量与场景参数 =====================
g        = 9.81;                 % 重力加速度 (m/s^2)
v_sink   = 3.0;                 % 云团起爆后下沉速度 (m/s)
R_cloud  = 10.0;                % 有效遮蔽半径 (m)
T_eff    = 20.0;                % 云团有效时长 (s)
decoy    = [0, 0, 0];           % 假目标（原点）
Tpt      = [0, 200, 5];         % 真目标代表点（圆柱中心高度，一阶近似，可改 0/10 做敏感性）
real_r   = 7.0;                 % 真目标圆柱半径
real_h   = 10.0;                % 真目标圆柱高度
U0       = [17800, 0, 1800];    % FY1 初始位置
M0       = [20000, 0, 2000];    % M1 初始位置
uM       = -M0 / norm(M0);      % M1 指向原点的单位方向
vM_vec   = 300 * uM;            % M1 速度向量（恒定）
phi0     = atan2(decoy(2)-U0(2), decoy(1)-U0(1));  % FY1 指向原点的基准角（此处约为 pi）

% 参数取值范围（可按需调整）
theta_lo = -pi;     theta_hi = +pi;     % 航向角偏移（相对 phi0），放宽到全向搜索
v_lo     = 70;      v_hi     = 140;     % FY1 速度范围 (m/s)
tr_lo    = 0.0;     tr_hi    = 20.0;    % 投放时刻范围 (s)（放宽上界以扩大搜索）
tf_lo    = 0.5;     tf_hi    = 10.0;    % 引信延迟范围 (s)（放宽上界以扩大搜索）

% 时间离散（粗/细两档）
dt_coarse = 0.02;    % 粗评估时间步
dt_fine   = 0.005;   % 微调/最终评估时间步

% 粗搜索策略（两选一）：随机采样 或 规则网格
use_random_coarse = true;   % true: 随机均匀采样；false: 规则网格
N_random          = 30000;  % 随机采样次数（建议 1e4~5e4）
% 若用规则网格，设定每维分辨率（注意组合数）
N_theta = 21;  % 航向离散数（例如 21 -> 步长约 π/10）
N_v     = 12;  % 速度离散数
N_tr    = 41;  % 投放时刻离散数
N_tf    = 41;  % 引信延迟离散数

% 目标几何：默认使用“多点覆盖阈值”以更贴近圆柱体（可切换）
use_multipoint = true;
p_thresh      = 1;     % 覆盖比例阈值（例如 90% 点被遮蔽即计入）
pts_cyl       = cylinder_points();

%% ===================== 粗搜索（可修改密度以控时） =====================
% 规则网格（若启用）或随机采样（默认）
if ~use_random_coarse
    theta_grid = linspace(theta_lo, theta_hi, N_theta);
    v_grid     = linspace(v_lo,     v_hi,     N_v    );
    tr_grid    = linspace(tr_lo,    tr_hi,    N_tr   );
    tf_grid    = linspace(tf_lo,    tf_hi,    N_tf   );
    % 组合数提示
    est_count = numel(theta_grid)*numel(v_grid)*numel(tr_grid)*numel(tf_grid);
    fprintf('规则网格估计组合数 ~ %d\n', est_count);
end

TopK = 10;   % 保留 Top-10 作为微调起点
cand  = [];  % 存 [J, theta, v, t_r, tau_f, x_e, y_e, z_e]

if use_random_coarse
    mode_str = '随机采样';
else
    mode_str = '规则网格';
end
fprintf('=== 粗搜索开始（%s）===\n', mode_str);
cnt = 0;
if use_random_coarse
    % 随机均匀采样
    for it = 1:N_random
        th = theta_lo + (theta_hi-theta_lo)*rand();
        v  = v_lo     + (v_hi    -v_lo    )*rand();
        tr = tr_lo    + (tr_hi   -tr_lo   )*rand();
        tf = tf_lo    + (tf_hi   -tf_lo   )*rand();
        phi = phi0 + th; hxy = [cos(phi), sin(phi)];
        te = tr + tf;
        xe = U0(1) + v * te * hxy(1);
        ye = U0(2) + v * te * hxy(2);
        ze = U0(3) - 0.5 * g * (tf^2);
        if use_multipoint
            J = score_hard_multi([xe,ye,ze], te, dt_coarse, M0, vM_vec, R_cloud, v_sink, T_eff, pts_cyl, p_thresh);
        else
            J = score_hard([xe,ye,ze], te, dt_coarse, M0, vM_vec, Tpt, R_cloud, v_sink, T_eff);
        end
        if isempty(cand)
            cand = [J, th, v, tr, tf, xe, ye, ze];
        else
            if size(cand,1) < TopK
                cand = [cand; J, th, v, tr, tf, xe, ye, ze]; %#ok<AGROW>
            else
                [Jmin, idxMin] = min(cand(:,1));
                if J > Jmin
                    cand(idxMin,:) = [J, th, v, tr, tf, xe, ye, ze];
                end
            end
        end
        cnt = cnt + 1;
    end
else
    % 规则网格扫描
    for th = theta_grid
        phi = phi0 + th;
        hxy = [cos(phi), sin(phi)];
        for v = v_grid
            for tr = tr_grid
                for tf = tf_grid
                    te = tr + tf;
                    xe = U0(1) + v * te * hxy(1);
                    ye = U0(2) + v * te * hxy(2);
                    ze = U0(3) - 0.5 * g * (tf^2);
                    if use_multipoint
                        J = score_hard_multi([xe,ye,ze], te, dt_coarse, M0, vM_vec, R_cloud, v_sink, T_eff, pts_cyl, p_thresh);
                    else
                        J = score_hard([xe,ye,ze], te, dt_coarse, M0, vM_vec, Tpt, R_cloud, v_sink, T_eff);
                    end
                    if isempty(cand)
                        cand = [J, th, v, tr, tf, xe, ye, ze];
                    else
                        if size(cand,1) < TopK
                            cand = [cand; J, th, v, tr, tf, xe, ye, ze]; %#ok<AGROW>
                        else
                            [Jmin, idxMin] = min(cand(:,1));
                            if J > Jmin
                                cand(idxMin,:) = [J, th, v, tr, tf, xe, ye, ze];
                            end
                        end
                    end
                    cnt = cnt + 1;
                end
            end
        end
    end
end
cand = sortrows(cand, -1);   % 按 J 降序
fprintf('粗网格完成：评估次数=%d，Top-1 粗评估遮蔽=%.3f s\n', cnt, cand(1,1));

%% ===================== 无导数局部微调（fminsearch + 平滑目标） =====================
% 采用参数映射：把无界变量 y ∈ R 通过 logistic 映射到物理区间，避免越界与罚项。
%   map(y; lo,hi) = lo + (hi-lo)/(1 + exp(-y))
% 以 TopK 候选为初值（把物理值反映回 y），做少量迭代微调，目标是“软遮蔽时长”的负值（最小化）
opts = optimset('Display','off', 'MaxFunEvals', 1500, 'MaxIter', 400);

best = struct('Jhard',-inf);
epsilon = 1.0;    % 软化阈值的平滑尺度（越小越接近硬判据，但优化更难）

for i = 1:min(TopK, size(cand,1))
    J0  = cand(i,1); th0 = cand(i,2); v0 = cand(i,3); tr0 = cand(i,4); tf0 = cand(i,5);
    % 把物理量反映回 y（逆 logistic）： y = log((x-lo)/(hi-x))
    invmap = @(x,lo,hi) log( (x-lo) / max(1e-12, (hi-x)) );
    y0 = [ invmap(th0, theta_lo, theta_hi), ...
           invmap(v0,  v_lo,     v_hi), ...
           invmap(tr0, tr_lo,    tr_hi), ...
           invmap(tf0, tf_lo,    tf_hi) ];

    obj = @(y) objective_soft_neg(y, ...
        theta_lo,theta_hi, v_lo,v_hi, tr_lo,tr_hi, tf_lo,tf_hi, ...
        phi0, U0, g, ...
        M0, vM_vec, Tpt, R_cloud, v_sink, T_eff, dt_fine, epsilon);

    yopt = fminsearch(obj, y0, opts);

    % 取最优 yopt 映射回物理量，并用硬判据做最终评估
    [theta1, v1, tr1, tf1] = map_vars(yopt, theta_lo,theta_hi, v_lo,v_hi, tr_lo,tr_hi, tf_lo,tf_hi);
    te1 = tr1 + tf1;
    phi1 = phi0 + theta1; h1 = [cos(phi1), sin(phi1)];
    xe1 = U0(1) + v1 * te1 * h1(1);
    ye1 = U0(2) + v1 * te1 * h1(2);
    ze1 = U0(3) - 0.5 * g * (tf1^2);
    if use_multipoint
        Jhard = score_hard_multi([xe1,ye1,ze1], te1, dt_fine, ...
                                 M0, vM_vec, R_cloud, v_sink, T_eff, pts_cyl, p_thresh);
    else
        Jhard = score_hard([xe1,ye1,ze1], te1, dt_fine, ...
                           M0, vM_vec, Tpt, R_cloud, v_sink, T_eff);
    end

    if Jhard > best.Jhard
        best.Jhard = Jhard;
        best.theta = theta1; best.v = v1; best.tr = tr1; best.tf = tf1; best.te = te1;
        best.E = [xe1,ye1,ze1];
    end
end

fprintf('=== 最优结果（硬判据） ===\n');
fprintf('航向偏移 theta = %.4f rad（相对 FY1->原点）\n', best.theta);
fprintf('速度 v          = %.2f m/s\n', best.v);
fprintf('投放时刻 t_r    = %.3f s\n', best.tr);
fprintf('引信延迟 tau_f  = %.3f s\n', best.tf);
fprintf('起爆时刻 t_e    = %.3f s\n', best.te);
fprintf('起爆点 E        = (%.3f, %.3f, %.3f) m\n', best.E(1), best.E(2), best.E(3));
fprintf('最大遮蔽时长     = %.3f s\n', best.Jhard);

%% ===================== 可视化：XY 俯视 + 距离曲线 =====================
% 生成最优解下的轨迹并画图（用与评估一致的时间步/判据）
dt_plot = dt_fine;
if use_multipoint
    [~, details] = score_hard_multi(best.E, best.te, dt_plot, ...
                                    M0, vM_vec, R_cloud, v_sink, T_eff, pts_cyl, p_thresh, true);
else
    [~, details] = score_hard(best.E, best.te, dt_plot, ...
                              M0, vM_vec, Tpt, R_cloud, v_sink, T_eff, true);
end
t      = details.t(:);
M      = details.M;
C      = details.C;
mask   = logical(details.mask(:));
d      = details.d(:);

% FY1 航迹（等高度直线），供近爆窗口与插图绘制
phi = phi0 + best.theta; h = [cos(phi), sin(phi), 0];
tU = linspace(0, best.te, max(2, ceil(best.te/dt_plot)));
U  = U0 + (tU(:) .* best.v) .* h; U(:,3)=U0(3);

% 终端打印遮蔽起止（若有多段则全部打印，去重消抖）
edges_v = diff([false; mask(:); false]);
idxS_v  = find(edges_v==1); idxE_v = find(edges_v==-1)-1;
% 消抖：若相邻区间起止时间差 < dt_plot/2，则合并为一个
merged = [];
for ii = 1:numel(idxS_v)
    ts = t(idxS_v(ii)); tei = t(idxE_v(ii));
    if isempty(merged), merged = [ts, tei]; continue; end
    if abs(ts - merged(end,1)) < (dt_plot/2) && abs(tei - merged(end,2)) < (dt_plot/2)
        merged(end,:) = [min(merged(end,1), ts), max(merged(end,2), tei)];
    else
        merged = [merged; ts, tei]; %#ok<AGROW>
    end
end
if isempty(merged)
    fprintf('【遮蔽起止】 无\n');
else
    % 打印一次综合起止（首段起点到末段终点），避免重复两行
    fprintf('【遮蔽起止】 [%.3f s, %.3f s]\n', merged(1,1), merged(end,2));
end

%% ===================== 本文件内联函数（评分与优化） =====================

function pts = cylinder_points()
% 生成圆柱体多点（顶/侧/底 + 中心）用于覆盖率判据
    cx=0; cy=200; r=7;
    th = linspace(0,2*pi,8+1); th(end)=[];
    pts = [ 0,200,5; ...
            cx+r*cos(th).', cy+r*sin(th).', 10*ones(8,1); ...
            cx+r*cos(th).', cy+r*sin(th).', 7.5*ones(8,1); ...
            cx+r*cos(th).', cy+r*sin(th).', 2.5*ones(8,1); ...
            cx+r*cos(th).', cy+r*sin(th).', zeros(8,1) ];
end

function [J, details] = score_hard_multi(E, te, dt, M0, vM_vec, R_cloud, v_sink, T_eff, pts, p_thresh, wantDetails)
% 多点覆盖阈值的硬判据：在 [te,te+T_eff] 内，按时刻统计被遮蔽点比例≥p_thresh 的时间积分
    if nargin < 11, wantDetails = false; end
    t = (te:dt:(te+T_eff)).'; Nt = numel(t);
    if Nt==0, J=0; details=[]; return; end
    M = M0 + t.*vM_vec;                           % Nt x 3
    C = [E(1)*ones(Nt,1), E(2)*ones(Nt,1), E(3) - v_sink*(t - te)];
    K = size(pts,1); masks = false(K,Nt);
    D = inf(K,Nt);                                  % 距离矩阵，用于可视化（取每时刻最小距离）
    for i=1:K
        Tpt = pts(i,:);
        AB = Tpt(:).' - M;                        % Nt x 3
        denom = sum(AB.^2, 2);
        numer = sum((C - M).*AB, 2);
        tau = zeros(Nt,1); nz = denom>eps; tau(nz) = numer(nz)./denom(nz);
        tau_c = min(max(tau,0),1);
        closest = M + tau_c.*AB;                  % Nt x 3（隐式扩展）
        di = sqrt(sum((C - closest).^2, 2));
        D(i,:) = di.';                            % 1xNt
        masks(i,:) = (di<=R_cloud) & (tau>=0) & (tau<=1);
    end
    frac = mean(masks,1);
    mask = frac >= p_thresh;
    J = sum(mask)*dt;
    if wantDetails
        dmin = min(D,[],1).';                      % Nt x 1（每个时刻所有点的最小距离）
        details = struct('t',t,'M',M,'C',C,'mask',mask,'frac',frac,'d',dmin);
    end
end

function [J, details] = score_hard(E, te, dt, M0, vM_vec, Tpt, R_cloud, v_sink, T_eff, wantDetails)
% 硬判据遮蔽时长：d(t)<=R_cloud 且 0<=tau<=1，在 [te, te+T_eff] 窗内积分
    if nargin < 10, wantDetails = false; end
    t = (te:dt:(te+T_eff)).';
    Nt = numel(t);
    % 轨迹（向量化）
    M = M0 + t.*vM_vec;            % Nt x 3
    C = [E(1)*ones(Nt,1), E(2)*ones(Nt,1), E(3) - v_sink*(t - te)];
    % 距离与段参数（向量化最近点）
    AB   = Tpt(:).' - M;            % Nt x 3
    denom= sum(AB.^2, 2);           % Nt x 1
    numer= sum((C - M).*AB, 2);     % Nt x 1
    tau  = zeros(Nt,1);
    nz   = denom > eps;
    tau(nz) = numer(nz)./denom(nz);
    tau_clamp = min(max(tau,0),1);
    closest = M + tau_clamp.*AB;    % Nt x 3 (隐式扩展)
    d = sqrt(sum((C - closest).^2, 2));
    mask = (d <= R_cloud) & (tau >= 0) & (tau <= 1);
    J = sum(mask)*dt;
    if wantDetails
        details = struct('t',t,'M',M,'C',C,'mask',mask,'d',d,'tau',tau);
    end
end

function Jneg = objective_soft_neg(y, ...
    theta_lo,theta_hi, v_lo,v_hi, tr_lo,tr_hi, tf_lo,tf_hi, ...
    phi0, U0, g, ...
    M0, vM_vec, Tpt, R_cloud, v_sink, T_eff, dt, epsilon)
% 软判据目标：平滑近似 I[d<=R]*I[0<=tau<=1] 的时间积分，取负用于最小化
    [theta, v, tr, tf] = map_vars(y, theta_lo,theta_hi, v_lo,v_hi, tr_lo,tr_hi, tf_lo,tf_hi);
    te = tr + tf;
    phi = phi0 + theta; h = [cos(phi), sin(phi)];
    E = [U0(1) + v*te*h(1), U0(2) + v*te*h(2), U0(3) - 0.5*g*(tf^2)];
    % 地表约束：起爆高度应非负，否则强惩罚
    if E(3) < 0
        Jneg = 1e6 + abs(E(3));
        return;
    end
    t = (te:dt:(te+T_eff)).';
    if isempty(t)
        Jneg = 0; return;
    end
    Nt = numel(t);
    M = M0 + t.*vM_vec;                                       % Nt x 3
    C = [E(1)*ones(Nt,1), E(2)*ones(Nt,1), E(3) - v_sink*(t - te)];
    AB   = Tpt(:).' - M;
    denom= sum(AB.^2, 2);
    numer= sum((C - M).*AB, 2);
    tau  = zeros(Nt,1); nz = denom>eps; tau(nz) = numer(nz)./denom(nz);
    tau_clamp = min(max(tau,0),1);
    closest = M + tau_clamp.*AB;
    d = sqrt(sum((C - closest).^2, 2));
    % 平滑指示函数（Logistic）
    sig = @(x) 1./(1+exp(-x));
    s_d   = sig((R_cloud - d)/epsilon);
    s_l0  = sig(tau/epsilon);
    s_l1  = sig((1 - tau)/epsilon);
    w     = s_d .* s_l0 .* s_l1;
    Jsoft = sum(w)*dt;
    Jneg  = -Jsoft;
end

function [theta, v, tr, tf] = map_vars(y, theta_lo,theta_hi, v_lo,v_hi, tr_lo,tr_hi, tf_lo,tf_hi)
% logistic 映射把无界 y 映射到物理区间 [lo, hi]
    map = @(yy,lo,hi) lo + (hi-lo)./(1 + exp(-yy));
    theta = map(y(1), theta_lo,theta_hi);
    v     = map(y(2), v_lo,    v_hi);
    tr    = map(y(3), tr_lo,   tr_hi);   % 反映“接到任务-反应延迟”，避免 0 s 角落解
    tf    = map(y(4), tf_lo,   tf_hi);
end