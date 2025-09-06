%% simulate_problem2_anim.m
% 题目A - 问题2：基于 p2.m 求得的最优方案进行三维仿真动画（中文注释）
% 说明：本脚本会读取 p2.m 运行时产生的最优解变量（best, U0, M0 等），
%       并生成类似问题1动画的可视化：导弹/无人机、释放/起爆点、云团球、LOS、视锥、遮蔽信息。

clear; clc; close all;

%% ============ 依赖最优解（请先运行 p2.m 获得 best 结构体） ============
% 为了复现最优解，这里直接调用 p2 脚本并保留工作区变量（需确保 p2.m 与本文件位于同一路径结构下）
run('p2.m');  % 运行后，工作区应有 best, U0, M0, vM_vec, Tpt, R_cloud, v_sink, T_eff, real_r, real_h 等

% 如果不想每次都重新优化，可把 best 的字段手动填入下方：
% best.theta = ...; best.v = ...; best.tr = ...; best.tf = ...; best.te = ...; best.E = [x_e,y_e,z_e];

%% ============ 常量与派生量 ============
g = 9.80665;                 % 重力加速度
decoy = [0,0,0];          % 假目标
real_center = [0,200,0];  % 真目标下底圆心

% 最优参数
theta = best.theta; vU = best.v; t_release = best.tr; t_fuze = best.tf; t_exp = best.te; Ept = best.E;
phi0 = atan2(decoy(2)-U0(2), decoy(1)-U0(1));
heading = [cos(phi0+deg2rad(theta)), sin(phi0+deg2rad(theta)), 0]; heading = heading ./ norm(heading);

% 释放点（用于标注，可通过最优解回推）
P_rel = U0 + (t_release * vU) * heading; P_rel(3) = U0(3);

%% ============ 动画时间轴 ============
t0 = 0.0; t1 = t_exp + T_eff; dt = 0.02; t = (t0:dt:t1).'; N = numel(t);

%% ============ 轨迹（数值） ============
% 导弹
M = M0 + t .* vM_vec;     % N x 3
% 无人机（等高度）
U = U0 + (t .* vU) .* heading; U(:,3) = U0(3);
% 云团中心（仅在 [t_exp, t_exp+T_eff] 有效）
C = nan(N,3);
idxCloud = (t >= t_exp) & (t <= t_exp + T_eff);
C(idxCloud,1) = Ept(1);
C(idxCloud,2) = Ept(2);
C(idxCloud,3) = Ept(3) - v_sink * (t(idxCloud) - t_exp);

%% ============ 预生成网格与容器 ============
[th, zz] = meshgrid(linspace(0,2*pi,60), linspace(0,real_h,8));
xc = real_center(1) + real_r * cos(th);
yc = real_center(2) + real_r * sin(th);
zc = zz;
[su, sv] = sphere(24);

% 预计算遮蔽度量（圆柱多点+覆盖阈值，与 p2.m 对齐）
R = R_cloud;
cx=0; cy=200; r=7; ths = linspace(0,2*pi,8+1); ths(end)=[];
pts_cyl = [ 0,200,5; ...
            cx+r*cos(ths).', cy+r*sin(ths).', 10*ones(8,1); ...
            cx+r*cos(ths).', cy+r*sin(ths).', 7.5*ones(8,1); ...
            cx+r*cos(ths).', cy+r*sin(ths).', 2.5*ones(8,1); ...
            cx+r*cos(ths).', cy+r*sin(ths).', zeros(8,1) ];
K = size(pts_cyl,1);
d = nan(N,1); mask = false(N,1);
p_thresh = 1.0; % 与 p2.m 对齐
for k = find(idxCloud).'
    Ck = C(k,:); Mk = M(k,:);
    covered = 0; dmin_k = inf;
    for i=1:K
        Tpi = pts_cyl(i,:);
        AB = Tpi - Mk; denom = dot(AB,AB);
        if denom < eps
            di = norm(Ck - Mk); taui = 0;
        else
            taui = dot(Ck - Mk, AB) / denom;
            tclamp = min(max(taui,0),1);
            Qi = Mk + tclamp * AB;
            di = norm(Ck - Qi);
        end
        if (di <= R) && (taui >= 0) && (taui <= 1)
            covered = covered + 1;
        end
        dmin_k = min(dmin_k, di);
    end
    mask(k) = (covered / K) >= p_thresh;
    d(k) = dmin_k; % 信息显示采用最小距离
end

edges = diff([false; mask; false]);
idxStart = find(edges == 1); idxEnd = find(edges == -1) - 1;
occlusion_total = sum(mask) * dt;

%% ============ 初始化图窗 ============
allXYZ = [M; U; decoy; Ept; [real_center(1),real_center(2),real_h]];
xlim3 = [min(allXYZ(:,1))-1500, max(allXYZ(:,1))+1500];
ylim3 = [min(allXYZ(:,2))-1500, max(allXYZ(:,2))+1500];
zlim3 = [0, max(allXYZ(:,3))+500];

fig = figure('Color','w','Position',[80 80 980 640]);
set(fig, 'Resize','off');
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); axis(ax,'equal');
xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Z (m)');
title(ax, '问题2：FY1-单弹最优方案（三维动画）');
xlim(ax, xlim3); ylim(ax, ylim3); zlim(ax, zlim3); view(ax, 30, 20);

% 静态要素
surf(ax, xc, yc, zc, 'EdgeAlpha',0.25,'FaceAlpha',0.15,'FaceColor',[0.3 0.6 1.0]);
plot3(ax, decoy(1),decoy(2),decoy(3), 'x', 'LineWidth',1.2,'MarkerSize',9);
plot3(ax, P_rel(1),P_rel(2),P_rel(3), 'ko','MarkerSize',8,'LineWidth',1.0);
plot3(ax, Ept(1), Ept(2), Ept(3),    'ks','MarkerSize',9,'LineWidth',1.0);

% 动态要素
hMpath = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2);
hUpath = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2);
hMnow  = plot3(ax, nan, nan, nan, '^', 'MarkerSize',8, 'MarkerFaceColor','auto');
hUnow  = plot3(ax, nan, nan, nan, 'o', 'MarkerSize',8, 'MarkerFaceColor','auto');
hLOS   = plot3(ax, nan, nan, nan, '-.', 'LineWidth',1.0);
hCloud = surf(ax, nan(2), nan(2), nan(2), 'EdgeAlpha',0.2, 'FaceAlpha',0.25);
hCone  = surf(ax, nan(2), nan(2), nan(2), 'EdgeAlpha',0.15, 'FaceAlpha',0.15, 'FaceColor',[1.0 0.5 0.5]);
hPerp  = plot3(ax, nan, nan, nan, ':', 'LineWidth',1.0, 'Color',[0.4 0.4 0.4]);
hDText = text(ax, 0,0,0, '', 'FontSize',9, 'Color',[0.2 0.2 0.2], ...
    'HorizontalAlignment','center','VerticalAlignment','bottom','Visible','off');

txtInfo = text(ax, xlim3(2), ylim3(2), zlim3(2), '', ...
    'HorizontalAlignment','right','VerticalAlignment','top','FontSize',10,'FontWeight','bold');

legend(ax, {'真目标(圆柱)','诱饵','释放点','起爆点','M1轨迹','FY1轨迹','M1当前位置','FY1当前位置','LOS(M1->真目标)','烟幕球'}, ...
    'Location','northeastoutside','FontSize',9);

% 视频输出
saveGIF = true; saveMP4 = true; gifName = 'output/problem2_anim.gif'; mp4Name = 'output/problem2_anim.mp4'; fps = 25;
if saveMP4
    vw = VideoWriter(mp4Name, 'MPEG-4'); vw.FrameRate = fps; open(vw);
end

% 尺寸偶数约束
targetH = []; targetW = [];

%% ============ 动画主循环 ============
for k = 1:N
    % 轨迹到当前
    set(hMpath, 'XData', M(1:k,1), 'YData', M(1:k,2), 'ZData', M(1:k,3));
    set(hUpath, 'XData', U(1:k,1), 'YData', U(1:k,2), 'ZData', U(1:k,3));
    set(hMnow,  'XData', M(k,1),   'YData', M(k,2),   'ZData', M(k,3));
    set(hUnow,  'XData', U(k,1),   'YData', U(k,2),   'ZData', U(k,3));

    % LOS
    set(hLOS, 'XData', [M(k,1), Tpt(1)], 'YData', [M(k,2), Tpt(2)], 'ZData', [M(k,3), Tpt(3)]);

    if idxCloud(k)
        Ck = C(k,:);
        set(hCloud, 'XData', R*su + Ck(1), 'YData', R*sv + Ck(2), 'ZData', R*su*0 + Ck(3) + R*sv*0, 'Visible','on');
        set(hLOS, 'Visible','on');
        if mask(k)
            set(hLOS, 'Color',[0 0.45 0.74], 'LineStyle','--');
        else
            set(hLOS, 'Color',[1 0 0], 'LineStyle','-');
        end

        % 垂线与数值
        Mk = M(k,:); AB = Tpt - Mk; denomAB = dot(AB,AB);
        if denomAB > eps
            tauk = dot(Ck - Mk, AB) / denomAB; tclamp = min(max(tauk,0),1); Q = Mk + tclamp * AB;
            set(hPerp, 'XData',[Ck(1) Q(1)], 'YData',[Ck(2) Q(2)], 'ZData',[Ck(3) Q(3)], 'Visible','on');
            midp = (Ck + Q)/2; set(hDText, 'Position', midp, 'String', sprintf('d=%.2f m', d(k)), 'Visible','on');
            if mask(k), set(hPerp,'Color',[0 0.45 0.74]); set(hDText,'Color',[0 0.45 0.74]);
            else,        set(hPerp,'Color',[1 0 0]);     set(hDText,'Color',[1 0 0]); end
        else
            set(hPerp, 'Visible','off'); set(hDText,'Visible','off');
        end

        % 视锥
        Mk = M(k,:); vCM = Ck - Mk; L = norm(vCM);
        if (L > R + 1e-6)
            alpha = asin(min(1, R / L)); e3 = vCM / L; tmp = [0 0 1]; if abs(dot(tmp,e3)) > 0.9, tmp = [0 1 0]; end
            e1 = cross(e3, tmp); e1 = e1 / norm(e1); e2 = cross(e3, e1);
            phi = linspace(0, 2*pi, 60);
            rdir = cos(alpha)*e3 + sin(alpha)*(cos(phi').*e1 + sin(phi').*e2);
            sceneSpan = max([diff(xlim3), diff(ylim3), diff(zlim3)]);
            Xc = [ (Mk(1) + 0.0*rdir(:,1))'; (Mk(1) + (L+2*sceneSpan)*rdir(:,1))' ];
            Yc = [ (Mk(2) + 0.0*rdir(:,2))'; (Mk(2) + (L+2*sceneSpan)*rdir(:,2))' ];
            Zc = [ (Mk(3) + 0.0*rdir(:,3))'; (Mk(3) + (L+2*sceneSpan)*rdir(:,3))' ];
            set(hCone, 'XData', Xc, 'YData', Yc, 'ZData', Zc, 'Visible','on');
        else
            set(hCone, 'Visible','off');
        end
    else
        set(hCloud, 'Visible','off'); set(hCone, 'Visible','off');
        set(hLOS, 'Visible','on', 'Color',[1 0 0], 'LineStyle','-');
        set(hPerp,  'Visible','off'); set(hDText,'Visible','off');
    end

    % 右上角信息
    occ = '否'; if ~isnan(d(k)) && mask(k), occ='是'; end
    dstr = 'N/A'; if ~isnan(d(k)), dstr = sprintf('%.2f m', d(k)); end
    cum_occ = sum(mask(1:k)) * dt;  % 累计遮蔽时长（实时）
    set(txtInfo, 'String', sprintf('t = %.3f s\n云团球心距LOS: %s\n遮蔽中: %s\n累计遮蔽: %.3f s', t(k), dstr, occ, cum_occ));

    drawnow;

    % 帧采集与偶数尺寸修正
    fr = getframe(fig); img = fr.cdata; [h,w,~] = size(img);
    if isempty(targetH), targetH = h - mod(h,2); targetW = w - mod(w,2); if targetH<=0, targetH=h; end; if targetW<=0, targetW=w; end; end
    out = zeros(targetH, targetW, 3, 'uint8'); out(1:min(targetH,h), 1:min(targetW,w), :) = img(1:min(targetH,h), 1:min(targetW,w), :);
    [A,map] = rgb2ind(out, 256);
    if saveGIF
        if k == 1, imwrite(A, map, 'problem2_anim.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
        else,      imwrite(A, map, 'problem2_anim.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps); end
    end
    if saveMP4, writeVideo(vw, out); end
end

if saveMP4, close(vw); end
fprintf('动画已保存：%s（GIF），%s（MP4）\n', 'problem2_anim.gif', 'problem2_anim.mp4');

%% d(t) 曲线与遮蔽窗口（并标注开始/结束时刻）
figure('Color','w','Position',[80 760 1180 360]); hold on; grid on;
plot(t, d, 'LineWidth',1.4);
yline(R,'--','R=10 m');
xline(t_exp,'-.','起爆');
xline(t_exp+T_eff,':','失效');
% 遮蔽区间阴影
yTop = max(R*2, max(d(~isnan(d)))*1.05);
for i = 1:numel(idxStart)
    xs = t(idxStart(i)); xe = t(idxEnd(i));
    patch([xs xe xe xs],[0 0 yTop yTop],[0.8 0.9 1.0],'FaceAlpha',0.35,'EdgeColor','none');
end
xlabel('时间 t (s)'); ylabel('云团中心到视线段LOS(M1->T)的距离 d(t) (m)');
title(sprintf('遮蔽总时长：%.3f s', occlusion_total));
xlim([t0, t1]); ylim([0, yTop]);
% 标注开始/结束遮蔽时间
for i = 1:numel(idxStart)
    xs = t(idxStart(i)); xe = t(idxEnd(i));
    xline(xs,'--',sprintf('开始 %.3f s', xs), ...
        'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left', ...
        'Color',[0 0.5 0],'LineWidth',1.0);
    xline(xe,':',sprintf('结束 %.3f s', xe), ...
        'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left', ...
        'Color',[0.5 0 0.5],'LineWidth',1.0);
end
