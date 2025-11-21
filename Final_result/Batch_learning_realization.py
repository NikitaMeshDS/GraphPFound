from enum import IntEnum
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from build_em_seq import build_em_sequences  # импорт из твоего файла
np.set_printoptions(precision=3, suppress=True)
from hmm_Prel_Plook_ref import (
    P_look as P_look_ref,
    P_rel as P_rel_ref,
    P_click as P_click_ref,
    P_break as P_break_ref,
)



class Color(IntEnum):
    GRAY = 0
    GREEN = 1
    YELLOW = 2


def generate_delivery(n_rows, n_cols, color_probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    colors = np.array(list(Color))
    probs = np.array([color_probs[c] for c in colors], dtype=float)
    probs /= probs.sum()

    idx = rng.choice(len(colors), size=(n_rows, n_cols), p=probs)
    layout_enum = colors[idx]
    return layout_enum


def get_transition_matrix(rel_matrix, p_look, delivery, p_break=0.15):
    """
    Строит матрицу переходов A_t для двух строк выдачи delivery[i:i+2, :].

    rel_matrix: shape (2, 2) – релевантность для этих двух строк
    p_look:     shape (n_colors, n_colors)
    delivery:   shape (2, 2) – цвета ячеек
    """
    delivery = np.asarray(delivery)
    colors = np.vectorize(int)(delivery)

    transition_matrix = np.zeros((13, 13))

    # верхняя левая ячейка
    p_move_right = p_look[colors[0, 0]][colors[0, 1]] / (
        p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]]
    )
    p_move_left_down = p_look[colors[0, 0]][colors[1, 0]] / (
        p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]]
    )

    transition_matrix[0, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down
    transition_matrix[0, 3] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break)
    transition_matrix[0, 4] = rel_matrix[0, 0]
    transition_matrix[0, 7] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1]
    transition_matrix[0, 8] = (1 - rel_matrix[0, 0]) * p_break
    transition_matrix[0, 11] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break

    transition_matrix[1, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down
    transition_matrix[1, 3] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break)
    transition_matrix[1, 4] = rel_matrix[0, 0]
    transition_matrix[1, 7] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1]
    transition_matrix[1, 8] = (1 - rel_matrix[0, 0]) * p_break
    transition_matrix[1, 11] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break

    # верхняя правая ячейка
    p_move_left = p_look[colors[0, 1]][colors[0, 0]] / (
        p_look[colors[0, 1]][colors[0, 0]] + p_look[colors[0, 1]][colors[1, 1]]
    )
    p_move_right_down = p_look[colors[0, 1]][colors[1, 1]] / (
        p_look[colors[0, 1]][colors[0, 0]] + p_look[colors[0, 1]][colors[1, 1]]
    )

    transition_matrix[2, 1] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break)
    transition_matrix[2, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down
    transition_matrix[2, 5] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0]
    transition_matrix[2, 6] = rel_matrix[0, 1]
    transition_matrix[2, 9] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break
    transition_matrix[2, 10] = (1 - rel_matrix[0, 1]) * p_break

    transition_matrix[3, 1] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break)
    transition_matrix[3, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down
    transition_matrix[3, 5] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0]
    transition_matrix[3, 6] = rel_matrix[0, 1]
    transition_matrix[3, 9] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break
    transition_matrix[3, 10] = (1 - rel_matrix[0, 1]) * p_break

    # абсорбирующее состояние
    transition_matrix[4:, 12] = 1.0

    return transition_matrix


def get_emission_matrix(matrix_click, p_break=0.15):
    """
    Эмиссионная матрица B_t для одной строки rel/click (shape (1, 2)).
    """
    transition_matrix = np.zeros((13, 8))
    pcL = matrix_click[0, 0]
    pcR = matrix_click[0, 1]

    transition_matrix[0, 0] = (1 - pcL)
    transition_matrix[0, 1] = pcL

    transition_matrix[1, 0] = (1 - pcL) * (1 - pcR)
    transition_matrix[1, 1] = (1 - pcL) * pcR
    transition_matrix[1, 2] = pcL * (1 - pcR)
    transition_matrix[1, 3] = pcL * pcR

    transition_matrix[2, 0] = (1 - pcR)
    transition_matrix[2, 2] = pcR

    transition_matrix[3, 0] = (1 - pcR) * (1 - pcL)
    transition_matrix[3, 1] = pcR * (1 - pcL)
    transition_matrix[3, 2] = (1 - pcR) * pcL
    transition_matrix[3, 3] = pcR * pcL

    transition_matrix[4, 0] = 1

    transition_matrix[5, 0] = (1 - pcL)
    transition_matrix[5, 2] = pcL

    transition_matrix[6, 0] = 1

    transition_matrix[7, 0] = (1 - pcR)
    transition_matrix[7, 1] = pcR

    transition_matrix[8, 4] = 1

    transition_matrix[9, 4] = (1 - pcL)
    transition_matrix[9, 6] = pcL

    transition_matrix[10, 5] = 1

    transition_matrix[11, 5] = (1 - pcR)
    transition_matrix[11, 7] = pcR

    transition_matrix[12, 0] = 1

    return transition_matrix


S_dict = {
    0: "S0", 1: "S1", 2: "S2", 3: "S3",
    4: "S0B", 5: "S1B", 6: "S2B", 7: "S3B",
    8: "S0D", 9: "S1D", 10: "S2D", 11: "S3D",
    12: "None"
}
X_dict = {0: "X0", 1: "X1", 2: "X2", 3: "X3", 4: "X4", 5: "X5", 6: "X6", 7: "X7"}


def get_start_states(rel_matrix, p_look, delivery, p_break=0.15):
    delivery = np.asarray(delivery)
    colors = np.vectorize(int)(delivery)
    start_states = np.zeros((13, 1))

    p_move_right = p_look[colors[0, 0]][colors[0, 1]] / (
        p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]]
    )
    p_move_left_down = p_look[colors[0, 0]][colors[1, 0]] / (
        p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]]
    )

    start_states[0, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down
    start_states[3, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break)
    start_states[4, 0] = rel_matrix[0, 0]
    start_states[7, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1]
    start_states[8, 0] = (1 - rel_matrix[0, 0]) * p_break
    start_states[11, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break

    return start_states


def e_step(pi, A_list, B_list, obs_seq):
    """
    Обычный forward–backward для нестационарной HMM:
    A_t, B_t зависят от t.
    """
    obs_seq = np.asarray(obs_seq, dtype=int)
    T = len(obs_seq)
    n_states = pi.shape[0]

    alpha = np.zeros((T, n_states))
    beta = np.zeros((T, n_states))
    gamma = np.zeros((T, n_states))
    xi = np.zeros((T - 1, n_states, n_states))

    # forward
    alpha[0] = pi * B_list[0][:, obs_seq[0]]
    for t in range(1, T):
        A_prev = A_list[t - 1]
        alpha[t] = (alpha[t - 1] @ A_prev) * B_list[t][:, obs_seq[t]]

    # backward
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        A_t = A_list[t]
        b_next = B_list[t + 1][:, obs_seq[t + 1]] * beta[t + 1]
        beta[t] = A_t @ b_next

    p_obs = np.dot(alpha[-1], beta[-1])
    log_likelihood = np.log(p_obs + 1e-300)

    # gamma
    for t in range(T):
        denom = np.dot(alpha[t], beta[t])
        gamma[t] = (alpha[t] * beta[t]) / (denom + 1e-300)

    # xi
    for t in range(T - 1):
        A_t = A_list[t]
        b_next = B_list[t + 1][:, obs_seq[t + 1]]
        numer = alpha[t][:, None] * A_t * (b_next * beta[t + 1])[None, :]
        denom = numer.sum()
        xi[t] = numer / (denom + 1e-300)

    return alpha, beta, gamma, xi, log_likelihood


def pack_params(rel_matrix, p_look, matrix_click, p_break):
    """
    Упаковка параметров в один вектор w.
    """
    return np.concatenate([
        rel_matrix.ravel(),
        p_look.ravel(),
        matrix_click.ravel(),
        np.array([p_break], dtype=float)
    ])


def unpack_params(w, rel_shape, p_look_shape, matrix_click_shape):
    """
    Обратная операция: из w достаем rel_matrix, p_look, matrix_click, p_break.
    Плюс клип к (1e-6, 1-1e-6) и нормировка строк p_look.
    """
    n_rel = int(np.prod(rel_shape))
    n_look = int(np.prod(p_look_shape))
    n_click = int(np.prod(matrix_click_shape))

    rel_flat = w[:n_rel]
    p_look_flat = w[n_rel:n_rel + n_look]
    click_flat = w[n_rel + n_look:n_rel + n_look + n_click]
    p_break = w[n_rel + n_look + n_click]

    rel_matrix = rel_flat.reshape(rel_shape)
    p_look = p_look_flat.reshape(p_look_shape)
    matrix_click = click_flat.reshape(matrix_click_shape)

    # клип, чтобы убрать точные 0 и 1
    rel_matrix = np.clip(rel_matrix, 1e-6, 1 - 1e-6)
    p_look = np.clip(p_look, 1e-6, 1 - 1e-6)
    matrix_click = np.clip(matrix_click, 1e-6, 1 - 1e-6)
    p_break = float(np.clip(p_break, 1e-6, 1 - 1e-6))

    # нормировка строк p_look
    p_look = p_look / p_look.sum(axis=1, keepdims=True)

    return rel_matrix, p_look, matrix_click, p_break

def neg_Q_batch(
    w,
    gamma_list,
    xi_list,
    obs_seqs,
    delivery,
    rel_shape,
    p_look_shape,
    matrix_click_shape,
    rel_prior,
    p_look_prior,
    matrix_click_prior,
    p_break_prior,
    lambda_rel=5.0,
    lambda_p_look=1.0,
    lambda_click=5.0,
    lambda_p_break=100.0,
):
    """
    Общий Q(w) для всего батча цепочек:
      Q(w) = sum_sess E[log p(O^{(s)}, Z^{(s)} | w)]  - регуляризация.

    gamma_list[k]  — гаммы для s-й цепочки, shape (T_k, n_states)
    xi_list[k]     — кси для s-й цепочки, shape (T_k-1, n_states, n_states)
    obs_seqs[k]    — наблюдения для s-й цепочки, длина T_k
    delivery       — общий layout выдачи (shape (T_max, 2))
    rel_shape, p_look_shape, matrix_click_shape — формы параметров.
    """
    # Распаковываем параметры
    rel_matrix, p_look, matrix_click, p_break = unpack_params(
        w,
        rel_shape,
        p_look_shape,
        matrix_click_shape
    )

    total_ll = 0.0

    # По всем сессиям
    for obs_seq, gamma, xi in zip(obs_seqs, gamma_list, xi_list):
        obs_seq = np.asarray(obs_seq, dtype=int)
        T = gamma.shape[0]
        n_states = gamma.shape[1]

        # По времени
        for t in range(T):
            # Эмиссии
            # Берём строку matrix_click[t], считаем B_t
            B_t = get_emission_matrix(
                matrix_click[t:t + 1, :],
                p_break=p_break
            )
            o_t = obs_seq[t]

            for i in range(n_states):
                val = B_t[i, o_t]
                if val > 1e-300:
                    total_ll += gamma[t, i] * np.log(val)

            # Переходы для t < T-1
            if t < T - 1:
                A_t = get_transition_matrix(
                    rel_matrix[t:t + 2, :],
                    p_look,
                    delivery[t:t + 2, :],
                    p_break=p_break
                )
                for i in range(n_states):
                    for j in range(n_states):
                        aij = A_t[i, j]
                        if aij > 1e-300:
                            total_ll += xi[t, i, j] * np.log(aij)

    # Регуляризация (одна на все сессии)
    reg = 0.0
    if rel_prior is not None:
        reg += lambda_rel * np.sum((rel_matrix - rel_prior) ** 2)
    if p_look_prior is not None:
        reg += lambda_p_look * np.sum((p_look - p_look_prior) ** 2)
    if matrix_click_prior is not None:
        reg += lambda_click * np.sum((matrix_click - matrix_click_prior) ** 2)
    reg += lambda_p_break * (p_break - p_break_prior) ** 2

    return -(total_ll - reg)

def em_batch_over_chains(
    rel_matrix_init,
    p_look_init,
    matrix_click_init,
    delivery,
    obs_seqs,
    pi,
    n_em_iters=10,
    lambda_rel=5.0,
    lambda_p_look=1.0,
    lambda_click=5.0,
    lambda_p_break=100.0,
    p_break_init=0.15,
    track_history=False,
):
    """
    Батч-EM: учимся по всем цепочкам obs_seqs сразу.

    rel_matrix_init : (T_max, 2)   — начальная матрица релевантностей (P_rel)
    p_look_init     : (3, 3)       — начальная матрица переходов по цветам
    matrix_click_init: (T_max, 2)  — начальная матрица кликов (P_click)
    delivery        : (T_max, 2)   — layout выдачи (цвета)
    obs_seqs        : список np.array, каждая длины T_k
    pi              : начальное распределение по 13 состояниям

    Если track_history=True — возвращает ещё и history с лоссом и параметрами.
    """
    rel_shape = rel_matrix_init.shape
    p_look_shape = p_look_init.shape
    click_shape = matrix_click_init.shape

    # Параметры-«якоря» для регуляризации
    rel_prior = rel_matrix_init.copy()
    p_look_prior = p_look_init.copy()
    matrix_click_prior = matrix_click_init.copy()
    p_break_prior = float(p_break_init)

    # Начальный вектор параметров
    w = pack_params(
        rel_matrix_init,
        p_look_init,
        matrix_click_init,
        p_break=p_break_init,
    )

    T_max = rel_matrix_init.shape[0]
    n_states = pi.shape[0]

    # История по итерациям EM
    history = {
        "loss": [],          # negQ_batch(w) = obj(w)
        "logL": [],          # суммарный log P(O | w) по всем цепочкам
        "rel": [],
        "p_look": [],
        "matrix_click": [],
        "p_break": [],
    }

    for it in range(n_em_iters):
        # ---------- E-шаг: считаем gamma, xi для всех цепочек ----------
        rel_matrix_cur, p_look_cur, matrix_click_cur, p_break_cur = unpack_params(
            w,
            rel_shape,
            p_look_shape,
            click_shape
        )

        gamma_list = []
        xi_list = []
        obs_eff_list = []
        total_logL = 0.0

        for o_seq in obs_seqs:
            if len(o_seq) == 0:
                continue

            # Обрезаем длину цепочки по максимально доступному T_max
            T_c = min(len(o_seq), T_max)
            obs_seq_c = np.asarray(o_seq[:T_c], dtype=int)

            # Строим A_t, B_t для этой цепочки
            A_list = []
            B_list = []
            for t in range(T_c - 1):
                A_list.append(
                    get_transition_matrix(
                        rel_matrix_cur[t:t + 2, :],
                        p_look_cur,
                        delivery[t:t + 2, :],
                        p_break=p_break_cur
                    )
                )
                B_list.append(
                    get_emission_matrix(
                        matrix_click_cur[t:t + 1, :],
                        p_break=p_break_cur
                    )
                )
            # последняя эмиссия
            B_list.append(
                get_emission_matrix(
                    matrix_click_cur[T_c - 1:T_c, :],
                    p_break=p_break_cur
                )
            )

            alpha, beta, gamma, xi, logL = e_step(pi, A_list, B_list, obs_seq_c)

            gamma_list.append(gamma)
            xi_list.append(xi)
            obs_eff_list.append(obs_seq_c)
            total_logL += logL

        print(f"[EM-batch] it={it}, total_logL(before M)={total_logL:.5f}")

        # ---------- M-шаг: максимизируем суммарный Q(w) ----------
        def obj(w_):
            return neg_Q_batch(
                w_,
                gamma_list=gamma_list,
                xi_list=xi_list,
                obs_seqs=obs_eff_list,
                delivery=delivery,
                rel_shape=rel_shape,
                p_look_shape=p_look_shape,
                matrix_click_shape=click_shape,
                rel_prior=rel_prior,
                p_look_prior=p_look_prior,
                matrix_click_prior=matrix_click_prior,
                p_break_prior=p_break_prior,
                lambda_rel=lambda_rel,
                lambda_p_look=lambda_p_look,
                lambda_click=lambda_click,
                lambda_p_break=lambda_p_break,
            )

        old_negQ = obj(w)
        res = minimize(
            fun=obj,
            x0=w,
            method="L-BFGS-B",
            options={"maxiter": 60}
        )

        w = res.x
        new_negQ = obj(w)
        print(f"         Q(old)={-old_negQ:.5f}, Q(new)={-new_negQ:.5f}")

        # Логируем историю после M-шагa
        if track_history:
            rel_new, p_look_new, click_new, p_break_new = unpack_params(
                w,
                rel_shape,
                p_look_shape,
                click_shape
            )
            history["loss"].append(new_negQ)
            history["logL"].append(total_logL)
            history["rel"].append(rel_new.copy())
            history["p_look"].append(p_look_new.copy())
            history["matrix_click"].append(click_new.copy())
            history["p_break"].append(p_break_new)

    # Распаковываем финальные параметры
    rel_opt, p_look_opt, matrix_click_opt, p_break_opt = unpack_params(
        w,
        rel_shape,
        p_look_shape,
        click_shape
    )

    if not track_history:
        return rel_opt, p_look_opt, matrix_click_opt, p_break_opt
    else:
        return rel_opt, p_look_opt, matrix_click_opt, p_break_opt, history


def compare_with_truth(rel_est, p_look_est, click_est, p_break_est,
                       n_rows_show=10):
    """
    rel_est, click_est: (T_used, 2)
    p_look_est: (3, 3)
    p_break_est: float
    Сравниваем с истинными матрицами из hmm_Prel_Plook_ref.
    """
    rel_true = np.array(P_rel_ref, dtype=float)
    click_true = np.array(P_click_ref, dtype=float)
    p_look_true = np.array(P_look_ref, dtype=float)
    p_break_true = float(P_break_ref)

    T_used = rel_est.shape[0]
    rel_true = rel_true[:T_used]
    click_true = click_true[:T_used]

    def stats(name, true, est, rows=None):
        if rows is not None:
            true = true[:rows]
            est = est[:rows]
        diff = est - true
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        max_abs = np.max(np.abs(diff))
        print(f"{name}:")
        print(f"  MSE     = {mse:.4f}")
        print(f"  MAE     = {mae:.4f}")
        print(f"  max|Δ|  = {max_abs:.4f}")
        print("  true:")
        print(np.round(true, 3))
        print("  est :")
        print(np.round(est, 3))
        print()

    print("\n===== СРАВНЕНИЕ С ИСТИННЫМИ ПАРАМЕТРАМИ =====")
    stats("P_rel (первые строки)", rel_true, rel_est, rows=n_rows_show)
    stats("P_click (первые строки)", click_true, click_est, rows=n_rows_show)
    stats("P_look (вся матрица)", p_look_true, p_look_est, rows=None)

    delta_p_break = p_break_est - p_break_true
    print(f"p_break: true = {p_break_true:.4f}, est = {p_break_est:.4f}, Δ = {delta_p_break:.4f}")
    print("=============================================\n")
    
def plot_batch_history(history, rel_true, click_true, p_look_true, p_break_true):
    """
    Рисует:
      1) лосс negQ_batch(w) по итерациям EM;
      2) нормы разности параметров от истинных.
    """
    loss = np.array(history["loss"])
    logL = np.array(history["logL"])  # на всякий случай, если пригодится
    rel_hist = history["rel"]
    p_look_hist = history["p_look"]
    click_hist = history["matrix_click"]
    p_break_hist = history["p_break"]

    iters = np.arange(len(loss))

    # --- нормы разности параметров ---
    norm_rel = [np.linalg.norm(r - rel_true) for r in rel_hist]
    norm_click = [np.linalg.norm(c - click_true) for c in click_hist]
    norm_look = [np.linalg.norm(L - p_look_true) for L in p_look_hist]
    norm_pbreak = [abs(pb - p_break_true) for pb in p_break_hist]

    norm_rel = np.array(norm_rel)
    norm_click = np.array(norm_click)
    norm_look = np.array(norm_look)
    norm_pbreak = np.array(norm_pbreak)

    # 1) Лосс (negQ) по итерациям
    plt.figure(figsize=(6, 4))
    plt.plot(iters, loss, marker="o")
    plt.xlabel("Итерация EM (batch)")
    plt.ylabel("negQ_batch(w)")
    plt.title("Лосс negQ_batch по итерациям batch-EM")
    plt.grid(True)

    # 2) Нормы разности параметров
    plt.figure(figsize=(7, 5))
    plt.plot(iters, norm_rel, label="‖P_rel - P_rel_true‖", marker="o")
    plt.plot(iters, norm_click, label="‖P_click - P_click_true‖", marker="o")
    plt.plot(iters, norm_look, label="‖P_look - P_look_true‖", marker="o")
    plt.plot(iters, norm_pbreak, label="|p_break - p_break_true|", marker="o")
    plt.xlabel("Итерация EM (batch)")
    plt.ylabel("Норма разности")
    plt.yscale("log")  # удобно смотреть динамику на лог-шкале
    plt.title("Сходимость параметров batch-EM")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def make_neg_Q(
    delivery,
    rel_shape,
    p_look_shape,
    matrix_click_shape,
    obs_seq,
    rel_prior,
    p_look_prior,
    matrix_click_prior,
    p_break_prior=0.15,
    # коэффициенты регуляризации (якорь к prior)
    lambda_rel=5.0,
    lambda_p_look=1.0,
    lambda_click=5.0,
    lambda_p_break=100.0,
):
    """
    Строит функцию neg_Q(w, gamma, xi) = –(E[log p(O,Z|w)] – регуляризация).

    Регуляризация — L2 относительно исходных параметров (prior),
    чтобы не уходить в дегenerate-углы.
    """
    obs_seq = np.asarray(obs_seq, dtype=int)

    def neg_Q(w, gamma, xi):
        T = gamma.shape[0]

        rel_matrix, p_look, matrix_click, p_break = unpack_params(
            w,
            rel_shape,
            p_look_shape,
            matrix_click_shape
        )

        total_ll = 0.0

        # сумма по времени
        for t in range(T):
            # эмиссии
            B_t = get_emission_matrix(
                matrix_click[t:t + 1, :],
                p_break=p_break
            )
            o_t = obs_seq[t]

            for i in range(gamma.shape[1]):
                val = B_t[i, o_t]
                if val > 1e-300:
                    total_ll += gamma[t, i] * np.log(val)

            if t == T - 1:
                break

            # переходы
            A_t = get_transition_matrix(
                rel_matrix[t:t + 2, :],
                p_look,
                delivery[t:t + 2, :],
                p_break=p_break
            )

            for i in range(gamma.shape[1]):
                for j in range(gamma.shape[1]):
                    aij = A_t[i, j]
                    if aij > 1e-300:
                        total_ll += xi[t, i, j] * np.log(aij)

        # L2-регуляризация вокруг prior
        reg = 0.0
        if rel_prior is not None:
            reg += lambda_rel * np.sum((rel_matrix - rel_prior) ** 2)
        if p_look_prior is not None:
            reg += lambda_p_look * np.sum((p_look - p_look_prior) ** 2)
        if matrix_click_prior is not None:
            reg += lambda_click * np.sum((matrix_click - matrix_click_prior) ** 2)
        reg += lambda_p_break * (p_break - p_break_prior) ** 2

        return -(total_ll - reg)

    return neg_Q



def em_with_tensor_AB(
    rel_matrix_init,
    p_look_init,
    matrix_click_init,
    delivery,
    obs_seq,
    pi,
    n_em_iters=10,
    lambda_rel=5.0,
    lambda_p_look=1.0,
    lambda_click=5.0,
    lambda_p_break=100.0,
    p_break_init=0.15,
    track_history=False,  # <- добавили флаг
):
    """
    EM: на E-шаге считаем gamma, xi для текущих A_t,B_t;
        на M-шаге численно максимизируем Q(w) – регуляризацию.

    Если track_history=True, дополнительно возвращаем историю:
      - loss (negQ)
      - logL
      - траекторию параметров (rel, p_look, matrix_click, p_break)
    """
    T = rel_matrix_init.shape[0]
    rel_shape = rel_matrix_init.shape
    p_look_shape = p_look_init.shape
    click_shape = matrix_click_init.shape

    # стартовый вектор параметров
    w = pack_params(
        rel_matrix_init,
        p_look_init,
        matrix_click_init,
        p_break=p_break_init,
    )

    # построили neg_Q с зафиксированными prior
    negQ = make_neg_Q(
        delivery=delivery,
        rel_shape=rel_shape,
        p_look_shape=p_look_shape,
        matrix_click_shape=click_shape,
        obs_seq=obs_seq,
        rel_prior=rel_matrix_init,
        p_look_prior=p_look_init,
        matrix_click_prior=matrix_click_init,
        p_break_prior=p_break_init,
        lambda_rel=lambda_rel,
        lambda_p_look=lambda_p_look,
        lambda_click=lambda_click,
        lambda_p_break=lambda_p_break,
    )

    # история (по итерациям EM)
    history = {
        "loss": [],          # negQ(w)
        "logL": [],          # log P(O|w)
        "rel": [],           # матрица релевантностей
        "p_look": [],        # матрица переходов по цветам
        "matrix_click": [],  # матрица кликов
        "p_break": [],       # скаляр p_break
    }

    for it in range(n_em_iters):
        # пересчитываем A_t,B_t для текущих параметров
        A_list = []
        B_list = []

        rel_matrix, p_look, matrix_click, p_break = unpack_params(
            w,
            rel_shape,
            p_look_shape,
            click_shape
        )

        for i in range(T - 1):
            A_list.append(
                get_transition_matrix(
                    rel_matrix[i:i + 2, :],
                    p_look,
                    delivery[i:i + 2, :],
                    p_break=p_break
                )
            )
            B_list.append(
                get_emission_matrix(
                    matrix_click[i:i + 1, :],
                    p_break=p_break
                )
            )

        # последняя строка для эмиссий
        B_list.append(
            get_emission_matrix(
                matrix_click[-1:], p_break=p_break
            )
        )

        # E-шаг
        alpha, beta, gamma, xi, logL = e_step(pi, A_list, B_list, obs_seq)

        Q_old = -negQ(w, gamma, xi)
        print(f"[EM] it={it}, logL(before M)={logL:.5f}, Q(old)={Q_old:.5f}")

        # M-шаг (численная оптимизация)
        res = minimize(
            fun=lambda w_: negQ(w_, gamma, xi),
            x0=w,
            method="L-BFGS-B",
            options={"maxiter": 60}
        )

        w = res.x
        Q_new = -negQ(w, gamma, xi)
        print(f"         Q(new)={Q_new:.5f}")

        # сохраняем историю после M-шагa
        if track_history:
            # текущие параметры
            rel_cur, p_look_cur, matrix_click_cur, p_break_cur = unpack_params(
                w,
                rel_shape,
                p_look_shape,
                click_shape
            )

            history["loss"].append(-Q_new)   # отрицательный Q (т.е. negQ)
            history["logL"].append(logL)
            history["rel"].append(rel_cur.copy())
            history["p_look"].append(p_look_cur.copy())
            history["matrix_click"].append(matrix_click_cur.copy())
            history["p_break"].append(p_break_cur)

    # проверка нормировки
    for t, A_t in enumerate(A_list):
        row_sums = A_t.sum(axis=1)
        print(f"A_list[{t}] row_sums (min={row_sums.min():.3f}, max={row_sums.max():.3f})")

    for t, B_t in enumerate(B_list):
        row_sums_B = B_t.sum(axis=1)
        print(f"B_list[{t}] row_sums (min={row_sums_B.min():.3f}, max={row_sums_B.max():.3f})")
        break

    rel_opt, p_look_opt, matrix_click_opt, p_break_opt = unpack_params(
        w,
        rel_shape,
        p_look_shape,
        click_shape
    )

    # если история не нужна — возвращаем как раньше (4 значения)
    if not track_history:
        return rel_opt, p_look_opt, matrix_click_opt, p_break_opt
    else:
        return rel_opt, p_look_opt, matrix_click_opt, p_break_opt, history
    

	


def train_sequential_on_chains(
    list_o,
    rel_matrix_full_init,
    matrix_click_full_init,
    P_look_init,
    delivery_full,
    p_break_init,
    n_em_iters_per_chain=5,
    min_len=2,
):
    """
    Последовательное EM-обучение по набору цепочек наблюдений.

    list_o                : список последовательностей наблюдений (список np.array или list[int])
    rel_matrix_full_init  : исходная rel-матрица (shape (n_rows, 2))
    matrix_click_full_init: исходная матрица кликов (shape (n_rows, 2))
    P_look_init           : матрица переходов по цветам (3 x 3)
    delivery_full         : матрица цветов выдачи (n_rows x 2)
    p_break_init          : исходное значение p_break

    n_em_iters_per_chain  : сколько EM-итераций делать на каждой цепочке
    min_len               : игнорировать цепочки короче этого порога
    """

    # Глобальные (текущие) параметры модели — будем их постепенно обновлять
    rel_global = rel_matrix_full_init.copy()
    click_global = matrix_click_full_init.copy()
    p_look_global = P_look_init.copy()
    p_break_global = float(p_break_init)

    n_states = 13
    n_rows = rel_global.shape[0]

    for idx, o_seq in enumerate(list_o):
        if len(o_seq) < min_len:
            # слишком короткая цепочка, forward-backward будет почти бессмысленен
            continue

        # длина цепочки в шагax HMM (ограничиваем сверху числом строк в rel_matrix)
        T_steps = min(len(o_seq), n_rows)
        obs_seq = np.array(o_seq[:T_steps], dtype=int)

        # ЛОКАЛЬНЫЕ параметры для этой цепочки — просто первые T_steps строк
        rel_local = rel_global[:T_steps].copy()
        click_local = click_global[:T_steps].copy()
        delivery_local = delivery_full[:T_steps].copy()

        # Начальное распределение pi — считаем из актуальных параметров
        start_vec = get_start_states(
            rel_local[:2, :],
            p_look_global,
            delivery_local[:2, :],
            p_break=p_break_global,
        )
        pi = start_vec.ravel().astype(float)
        s = pi.sum()
        if s > 0:
            pi /= s
        else:
            pi = np.full(n_states, 1.0 / n_states)

        print(f"\n=== EM по цепочке #{idx}, длина {T_steps} ===")

        # Запускаем EM для ОДНОЙ ЦЕПОЧКИ, но стартуем с текущих глобальных параметров
        rel_opt, p_look_opt, click_opt, p_break_opt = em_with_tensor_AB(
            rel_matrix_init=rel_local,
            p_look_init=p_look_global,
            matrix_click_init=click_local,
            delivery=delivery_local,
            obs_seq=obs_seq,
            pi=pi,
            n_em_iters=n_em_iters_per_chain,
            p_break_init=p_break_global,
        )

        # ОБНОВЛЯЕМ глобальные параметры тем, что выучили на этой цепочке
        rel_global[:T_steps] = rel_opt
        click_global[:T_steps] = click_opt
        p_look_global = p_look_opt
        p_break_global = p_break_opt

    return rel_global, p_look_global, click_global, p_break_global



def demo_batch(n_em_iters=5, n_chains=10, n_pairs=20):
    """
    Демо батч-EM:
      1) берём истинные P_rel, P_click, P_look, P_break как init/«prior»;
      2) генерим delivery;
      3) генерим n_chains сессий через build_em_sequences;
      4) учимся по всем цепочкам сразу;
      5) сравниваем результат с истиной.
    """
    # Истинные матрицы
    rel_matrix_full = np.array(P_rel_ref, dtype=float)[:n_pairs]
    matrix_click_full = np.array(P_click_ref, dtype=float)[:n_pairs]
    P_look = np.array(P_look_ref, dtype=float)
    p_break_true = float(P_break_ref)

    print("rel_matrix_full.shape =", rel_matrix_full.shape)

    # Генерим layout выдачи (цвета)
    color_probs = {
        Color.GRAY: 0.5,
        Color.GREEN: 0.3,
        Color.YELLOW: 0.2,
    }
    delivery_full = generate_delivery(
        n_rows=rel_matrix_full.shape[0],
        n_cols=2,
        color_probs=color_probs,
    )
    print("delivery_full.shape   =", delivery_full.shape)

    # Генерируем все цепочки
    list_z, list_o = build_em_sequences(
        n_chains=n_chains,
        n_pairs=n_pairs,
    )

    lengths = [len(o) for o in list_o]
    print("\n--- Сводка по сгенерированным последовательностям ---")
    print("Длины o_seq по цепочкам:", lengths[:20], "...")
    print("Суммарное число наблюдений:", sum(lengths))

    # Начальное распределение pi считаем по первой паре документов
    start_vec = get_start_states(
        rel_matrix_full[0:2, :],
        P_look,
        delivery_full[0:2, :],
        p_break=p_break_true,
    )
    n_states = 13
    pi = start_vec.ravel().astype(float)
    s = pi.sum()
    if s > 0:
        pi /= s
    else:
        pi = np.full(n_states, 1.0 / n_states)

    print("\nНачальное распределение pi:\n", np.round(pi, 4))

    # Запускаем батч-EM
        # Запускаем батч-EM с логированием истории
    rel_opt, p_look_opt, matrix_click_opt, p_break_opt, history = em_batch_over_chains(
        rel_matrix_init=rel_matrix_full,
        p_look_init=P_look,
        matrix_click_init=matrix_click_full,
        delivery=delivery_full,
        obs_seqs=list_o,
        pi=pi,
        n_em_iters=n_em_iters,
        lambda_rel=5.0,
        lambda_p_look=1.0,
        lambda_click=5.0,
        lambda_p_break=100.0,
        p_break_init=p_break_true,
        track_history=True,
    )


    print("\n===== ИТОГ после батч-EM по всем цепочкам =====")
    print("rel_matrix (первые 10 строк):")
    print(np.round(rel_opt[:10], 3))
    print("\np_look:")
    print(np.round(p_look_opt, 3))
    print("\nmatrix_click (первые 10 строк):")
    print(np.round(matrix_click_opt[:10], 3))
    print("\np_break =", p_break_opt)

    # Сравнение с истиной
    compare_with_truth(
        rel_est=rel_opt,
        p_look_est=p_look_opt,
        click_est=matrix_click_opt,
        p_break_est=p_break_opt,
        n_rows_show=10,
    )
        # Графики сходимости batch-EM
    plot_batch_history(
        history,
        rel_true=rel_matrix_full,
        click_true=matrix_click_full,
        p_look_true=P_look,
        p_break_true=p_break_true,
    )




if __name__ == "__main__":
    # Старый demo при желании можешь оставить:
    # demo(5, sostoyanie=0, n_chains=5, n_pairs=20)

    # Новый батч-EM по всем цепочкам
    demo_batch(n_em_iters=10, n_chains=10, n_pairs=20)
