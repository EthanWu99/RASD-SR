import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score


def evaluate_anomaly_scores_multi(
        folder_paths,
        output_dir='output_results',
        idx=1,
        draw_plot=False,
        save_result=True
):
    os.makedirs(output_dir, exist_ok=True)
    official_score = []

    for ww in range(idx):
        AUC, pAUC, Mean_source, Mean_target = [], [], [], []

        for target_string in ['anomaly_score_bearing', 'anomaly_score_fan', 'anomaly_score_gearbox',
                              'anomaly_score_slider', 'anomaly_score_ToyCar', 'anomaly_score_ToyTrain',
                              'anomaly_score_valve']:
            all_data = []
            file = None

            # 遍历所有文件夹
            for folder_path in folder_paths:
                for file_name in os.listdir(folder_path):
                    if target_string in file_name and file_name.endswith('.csv'):
                        full_path = os.path.join(folder_path, file_name)
                        df = pd.read_csv(full_path, header=None)
                        column_data = df.iloc[:, 1]
                        norm_data = (column_data - column_data.min()) / (column_data.max() - column_data.min())
                        all_data.append(norm_data)
                        if file is None:
                            file = df.iloc[:, 0]

            if not all_data:
                print(f"⚠️ No data found for {target_string}")
                continue

            all_data_df = pd.concat(all_data, axis=1)
            mean_values = all_data_df.mean(axis=1)
            all_data_np = np.array(all_data)

            mean_values = (mean_values - mean_values.min()) / (mean_values.max() - mean_values.min())
            std_values = np.var(all_data_np, axis=0)
            std_values = (std_values - std_values.min()) / (std_values.max() - std_values.min())

            w = ww / 1000
            abnormal_score = (1 - w) * mean_values + w * std_values

            # 构造标签
            y_true = np.zeros(200)
            y_true[100:] = 1
            source = np.concatenate((abnormal_score[:50], abnormal_score[100:150]))
            target = np.concatenate((abnormal_score[50:100], abnormal_score[150:]))
            source_true = np.zeros(100)
            source_true[50:] = 1
            target_true = np.zeros(100)
            target_true[50:] = 1

            # 指标评估
            auc = roc_auc_score(y_true, abnormal_score)
            pauc = roc_auc_score(y_true, abnormal_score, max_fpr=0.1)
            source_auc = roc_auc_score(source_true, source)
            target_auc = roc_auc_score(target_true, target)

            OS = hmean([target_auc, source_auc, pauc])
            print(
                f'{target_string}\tOS: {OS:.4f}\tpAUC: {pauc:.4f}\tsource_AUC: {source_auc:.4f}\ttarget_AUC: {target_auc:.4f}')

            AUC.append(auc)
            pAUC.append(pauc)
            Mean_source.append(source_auc)
            Mean_target.append(target_auc)

            # 保存融合得分为 CSV 文件
            if save_result:
                result_df = pd.DataFrame({'File Name': file, 'Mean Value': abnormal_score})
                save_name = f"{target_string}_ensemble_w{w:.1f}.csv"
                result_path = os.path.join(output_dir, save_name)
                result_df.to_csv(result_path, index=False, header=False)

        # 综合得分计算
        if len(AUC) == 0:
            continue
        mean_p_auc = hmean(pAUC)
        mean_source_auc = hmean(Mean_source)
        mean_target_auc = hmean(Mean_target)
        final_score = hmean([mean_source_auc, mean_target_auc, mean_p_auc])
        print(
            f'Overall (w={w:.1f})\tOfficial Score: {final_score:.4f}\tSource: {mean_source_auc:.4f}\tTarget: {mean_target_auc:.4f}\tpAUC: {mean_p_auc:.4f}')
        official_score.append(final_score)

    # 可视化
    if draw_plot and idx > 1:
        x = np.arange(idx)
        y = np.array(official_score)
        max_idx = np.argmax(y)
        max_x = x[max_idx]
        max_y = y[max_idx]

        fig, ax = plt.subplots()
        ax.plot(x, y, label='Score', color='blue')
        ax.plot(max_x, max_y, 'ro')
        ax.annotate(f'Max: ({max_x}, {max_y:.4f})', xy=(max_x, max_y),
                    xytext=(max_x + 0.5, max_y), arrowprops=dict(facecolor='black'), fontsize=12)
        ax.set_title('Score vs Weight w')
        ax.set_xlabel('w (weight index / 10)')
        ax.set_ylabel('Harmonic Mean Score')
        ax.grid(True)
        ax.legend()
        plt.show()

    return official_score


def ds_fusion_bpa(scores):
    """对一组模型输出的异常得分进行 DS 融合，返回异常概率"""
    m = []
    for s in scores:
        s = np.clip(s, 1e-6, 1 - 1e-6)
        m.append({'normal': 1 - s, 'anomaly': s})
    fused = m[0]
    for i in range(1, len(m)):
        a, b = fused, m[i]
        K = a['normal'] * b['anomaly'] + a['anomaly'] * b['normal']
        if K >= 1:
            fused = {'normal': 0.5, 'anomaly': 0.5}
        else:
            normal = (a['normal'] * b['normal']) / (1 - K)
            anomaly = (a['anomaly'] * b['anomaly']) / (1 - K)
            fused = {'normal': normal, 'anomaly': anomaly}
    return fused['anomaly']


def calculate_auc_pauc(fused_scores, y_true):
    auc = roc_auc_score(y_true, fused_scores)
    pauc = roc_auc_score(y_true, fused_scores, max_fpr=0.1)
    return auc, pauc


def ds_fusion_matrix(data, num):
    assert data.shape[1] % num == 0, "列数必须是5的倍数"
    n_group = data.shape[1] // num
    n_row = data.shape[0]

    result = np.zeros((n_row, num))

    for pos in range(num):  # 遍历每个位置
        for i in range(n_row):
            # 收集每组中的该位置的值
            values = [data[i, pos + num * g] for g in range(n_group)]
            # 进行 DS 融合
            result[i, pos] = ds_fusion_bpa(values)

    return result


def ds_fusion_matrix_avg_then_ds(data, num, weights):
    assert data.shape[1] == 5 * num, "列数应为5*num"
    N = data.shape[0]
    result = np.zeros(N)

    for i in range(N):
        row = data[i] * weights
        row_reshaped = row.reshape(num, 5)  # shape: (num, 5)
        row_mean = np.average(row_reshaped, axis=1)  # shape: (num,)
        result[i] = ds_fusion_bpa(row_mean)

    return result


def is_name_matched(base_names, current_names):
    return all(any(b in c or c in b for c in current_names) for b in base_names)


def fuse_all_categories_ds(folder_paths, category_names, save_dir, DS=False, weights=None, printBool=True, iseval=0):
    os.makedirs(save_dir, exist_ok=True)
    result_dict = {}

    for category in category_names:
        all_scores = []
        file_names = None

        for folder in folder_paths:
            file_list = [f for f in os.listdir(folder) if category in f and f.endswith('.csv')]
            for file in file_list:
                path = os.path.join(folder, file)
                df = pd.read_csv(path, header=None)

                current_names = df.iloc[:, 0].astype(str).apply(os.path.basename).tolist()
                scores_raw = df.iloc[:, 1].tolist()
                name_to_score = dict(zip(current_names, scores_raw))

                if file_names is None:
                    file_names = current_names
                    scores = scores_raw
                else:
                    try:
                        scores = [name_to_score[os.path.basename(name)] for name in file_names]
                    except KeyError as e:
                        print(f"[跳过] 文件 {file} 中缺失名称: {e}")
                        continue

                all_scores.append(scores)

        if len(all_scores) < 2:
            print(f"[警告] 类别 {category} 的可用文件不足两个，跳过。")
            continue

        all_scores = np.array(all_scores).T  # [样本数, 模型数]

        # 归一化到 [0, 1]
        all_scores = (all_scores - all_scores.min(axis=0)) / (all_scores.max(axis=0) - all_scores.min(axis=0))

        if DS:
            if weights is None:
                # fused_scores = ds_fusion_matrix_avg_then_ds(all_scores, num=len(folder_paths), weights=weights)
                a = ds_fusion_matrix(all_scores, num=5)
                fused_scores = np.mean(a, axis=1)
            else:
                weights = np.array(weights)
                weights = weights / (np.sum(weights))
                if weights.shape[0] != all_scores.shape[1]:
                    raise ValueError(f"权重数量 {len(weights)} 与模型数量 {all_scores.shape[1]} 不一致")
                fused_scores = ds_fusion_matrix_avg_then_ds(all_scores, num=len(folder_paths), weights=weights)
        else:
            if weights is None:
                fused_scores = np.mean(all_scores, axis=1)
            else:
                weights = np.array(weights)
                weights = weights / (np.sum(weights))
                if weights.shape[0] != all_scores.shape[1]:
                    raise ValueError(f"权重数量 {len(weights)} 与模型数量 {all_scores.shape[1]} 不一致")
                fused_scores = np.average(all_scores, axis=1, weights=weights)

        if iseval == 1:
            y_true = np.array(pd.read_csv(
                './dcase2023_task2_evaluator-main/ground_truth_data/ground_truth_' + category.split('_')[
                    2] + '_section_00' + '_test.csv', header=None).iloc[:, 1] == 1)

            source_all = np.array(pd.read_csv(
                './dcase2023_task2_evaluator-main/ground_truth_domain/ground_truth_' + category.split('_')[
                    2] + '_section_00' + '_test.csv', header=None).iloc[:, 1] == 0)
            source_AUC = roc_auc_score(y_true[source_all], fused_scores[source_all])
            target_AUC = roc_auc_score(y_true[~source_all], fused_scores[~source_all])
        else:
            y_true = np.zeros(len(fused_scores))
            y_true[100:] = 1

            # 计算源和目标 AUC
            data1 = fused_scores[:50]
            data2 = fused_scores[50:100]
            data3 = fused_scores[100:150]
            data4 = fused_scores[150:]
            source = np.concatenate((data1, data3))
            target = np.concatenate((data2, data4))
            source_true = np.zeros(100)
            source_true[50:] = 1
            target_true = np.zeros(100)
            target_true[50:] = 1
            source_AUC = roc_auc_score(source_true, source)
            target_AUC = roc_auc_score(target_true, target)

        auc, pauc = calculate_auc_pauc(fused_scores, y_true)

        # 保存融合结果
        save_path = os.path.join(save_dir, f'{category}_section_00_test.csv')
        result_df = pd.DataFrame({'File Name': file_names, 'Fused Score': fused_scores})
        result_df.to_csv(save_path, index=False, header=False)

        result_dict[category] = {
            'fused_scores': fused_scores,
            'AUC': auc,
            'pAUC': pauc,
            'source_AUC': source_AUC,
            'target_AUC': target_AUC
        }

        if printBool == True:
            OS = hmean([target_AUC, source_AUC, pauc])
            print(
                f'{category}\tOS: {OS:.4f}\tpAUC: {pauc:.4f}\tsource_AUC: {source_AUC:.4f}\ttarget_AUC: {target_AUC:.4f}')

    # 汇总整体评价
    all_auc = [result['AUC'] for result in result_dict.values()]
    all_pauc = [result['pAUC'] for result in result_dict.values()]
    mean_auc = hmean(all_auc) if len(all_auc) > 1 else np.mean(all_auc)
    mean_pauc = hmean(all_pauc) if len(all_pauc) > 1 else np.mean(all_pauc)
    all_source_AUC = [result['source_AUC'] for result in result_dict.values()]
    all_target_AUC = [result['target_AUC'] for result in result_dict.values()]
    mean_source_AUC = hmean(all_source_AUC) if len(all_source_AUC) > 1 else np.mean(all_source_AUC)
    mean_target_AUC = hmean(all_target_AUC) if len(all_target_AUC) > 1 else np.mean(all_target_AUC)
    final_score = hmean([mean_source_AUC, mean_target_AUC, mean_pauc])

    if printBool == True:
        print(
            f"Overall\tOfficial Score: {final_score:.4f}\tSource: {mean_source_AUC:.4f}\tTarget: {mean_target_AUC:.4f}\tpAUC: {mean_pauc:.4f}")

    return result_dict, final_score


def run_single_combination(folders, weights, category_names, save_dir, DS, iseval=0):
    _, final_score = fuse_all_categories_ds(folders, category_names, save_dir, DS=DS, weights=np.repeat(weights, 5),
                                            iseval=iseval)
    return final_score


def heuristic_search(folders_all, category_names, save_dir, DS=True,
                     search_weights=True, search_combinations=True,
                     initial_weights=None, initial_combination=None,
                     max_iter=20, perturb_scale=0.1, decay_rate=0.95, iseval=0):

    if initial_combination is None:
        initial_combination = list(range(len(folders_all)))
    current_combination = initial_combination.copy()

    if initial_weights is None:
        initial_weights = np.ones(len(current_combination)) / len(current_combination)
    else:
        initial_weights = initial_weights[::5]
    current_weights = initial_weights.copy()

    best_score = -np.inf
    best_combination = current_combination.copy()
    best_weights = current_weights.copy()

    history = []

    def eval_combo_weights(combo, weights, iseval=0):
        folders = [folders_all[i] for i in combo]
        w = weights / weights.sum()
        score = run_single_combination(folders, w, category_names, save_dir, DS, iseval=iseval)
        return score

    initial_perturb_scale = perturb_scale

    for it in range(max_iter):
        improved = False
        perturb_scale = initial_perturb_scale * (decay_rate ** it)

        if search_combinations:
            candidate_combos = []

            if len(current_combination) > 1:
                for rm_idx in current_combination:
                    new_combo = [i for i in current_combination if i != rm_idx]
                    candidate_combos.append(new_combo)

            not_selected = [i for i in range(len(folders_all)) if i not in current_combination]
            for add_idx in not_selected:
                new_combo = current_combination + [add_idx]
                candidate_combos.append(new_combo)

            for combo in candidate_combos:
                weights = np.ones(len(combo)) / len(combo)
                score = eval_combo_weights(combo, weights, iseval=iseval)
                if score > best_score:
                    best_score = score
                    best_combination = combo
                    best_weights = weights
                    improved = True

            if improved:
                current_combination = best_combination.copy()
                current_weights = best_weights.copy()

        if search_weights:
            for _ in range(5):
                noise = np.random.uniform(-perturb_scale, perturb_scale, size=len(current_weights))
                candidate_weights = current_weights + noise
                candidate_weights = np.clip(candidate_weights, 0.01, None)  # 防止为0或负
                candidate_weights /= candidate_weights.sum()

                score = eval_combo_weights(current_combination, candidate_weights, iseval=iseval)
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights
                    improved = True

            if improved:
                current_weights = best_weights.copy()

        history.append((best_score, best_combination.copy(), best_weights.copy()))
        print(f"Iter {it + 1}: Best Score={best_score:.4f}, Combination={best_combination}, Weights={best_weights}")

        if not improved:
            print("No improvement this iteration, stopping early.")
            # break
    return best_combination, best_weights, best_score, history


def run_multiple_heuristic_searches(
        num_runs, folders_all, category_names, save_dir,
        DS=False, search_weights=True, search_combinations=False,
        initial_weights=None, initial_combination=None,
        max_iter=20, perturb_scale=0.1, decay_rate=0.95,
        plot_title="Heuristic Search Stability", iseval=0
):
    results = []
    plt.figure(figsize=(10, 6))

    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        best_combination, best_weights, best_score, history = heuristic_search(
            folders_all, category_names, save_dir,
            DS=DS,
            search_weights=search_weights,
            search_combinations=search_combinations,
            initial_weights=initial_weights,
            initial_combination=initial_combination,
            max_iter=max_iter,
            perturb_scale=perturb_scale,
            decay_rate=decay_rate,
            iseval=iseval
        )
        results.append((best_combination, best_weights, best_score, history))

        scores = [h[0] for h in history]
        plt.plot(range(1, len(scores) + 1), scores, marker='*', label=f"No. {run + 1}")

    plt.title(plot_title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    folders = [
        # BEATs
        # r'D:/forth/ASD/DCASE2025/ResNet2025/teams/submission/AKNN_2023_wik/',
        # r'D:/forth/ASD/DCASE2025/pretrain/teams/submission/AKNN_2023_BEATs/',

        # r'D:/forth/ASD/DCASE2025/ResNet2025/teams/submission/AKNN_2024_wik/',
        # r'D:/forth/ASD/DCASE2025/pretrain/teams/submission/AKNN_2024_BEATs/',

        # r'D:/forth/ASD/DCASE2025/ResNet2025/teams/submission/AKNN_2025_wik/',
        # r'D:/forth/ASD/DCASE2025/pretrain/teams/submission/AKNN_2025_BEATs/',

    ]

    # 用以提交系统一(所有网络均使用，平均)
    weights = None

    category_names = [
        'anomaly_score_bearing', 'anomaly_score_fan', 'anomaly_score_gearbox',
        'anomaly_score_slider', 'anomaly_score_ToyCar', 'anomaly_score_ToyTrain', 'anomaly_score_valve'
    ]

    category_names_eval23 = [
        'anomaly_score_bandsaw', 'anomaly_score_grinder', 'anomaly_score_shaker',
        'anomaly_score_ToyDrone', 'anomaly_score_ToyNscale', 'anomaly_score_ToyTank', 'anomaly_score_Vacuum',
    ]

    category_names_eval24 = [
        'anomaly_score_3DPrinter', 'anomaly_score_AirCompressor', 'anomaly_score_BrushlessMotor',
        'anomaly_score_HairDryer', 'anomaly_score_HoveringDrone', 'anomaly_score_RoboticArm', 'anomaly_score_Scanner',
        'anomaly_score_ToothBrush', 'anomaly_score_ToyCircuit',
    ]

    category_names_eval25 = [
        'anomaly_score_AutoTrash', 'anomaly_score_BandSealer', 'anomaly_score_CoffeeGrinder',
        'anomaly_score_HomeCamera', 'anomaly_score_Polisher', 'anomaly_score_ScrewFeeder', 'anomaly_score_ToyPet',
        'anomaly_score_ToyRCCar'
    ]
    weights_dev = np.repeat([0.06229572, 0.01786699, 0.10848947, 0.19332208, 0.01023307, 0.05513917,
                             0.09599214, 0.01023307, 0.20231073, 0.19703107, 0.01330007, 0.02355335,
                             0.01023307, ], 5)

    weights_eval = np.repeat([0.0082083, 0.09171108, 0.08310833, 0.19963402, 0.14673729, 0.18281166,
                              0.0645916, 0.09915742, 0.0082083, 0.0082083, 0.0082083, 0.06457841,
                              0.034837, ], 5)
    save_dir = r'./fusion_results'

    print('################average############################')
    ensemble_results, final_score = fuse_all_categories_ds(folders, category_names, save_dir, DS=False, weights=weights)
    ensemble_results, final_score = fuse_all_categories_ds(folders, category_names_eval23, save_dir, DS=False,
                                                           weights=weights, iseval=1)


    print('################ACP############################')
    results = run_multiple_heuristic_searches(
        num_runs=5,
        folders_all=folders,
        category_names=category_names,
        initial_weights=weights,
        save_dir='search_logs',
        max_iter=30,
        perturb_scale=0.2,
        decay_rate=0.95,
        iseval=0
    )

    results_eval = run_multiple_heuristic_searches(
        num_runs=5,
        folders_all=folders,
        category_names=category_names_eval23,
        initial_weights=weights,
        save_dir='search_logs',
        max_iter=30,
        perturb_scale=0.2,
        decay_rate=0.95,
        iseval=1
    )
