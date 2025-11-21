from enum import IntEnum
import numpy as np
from scipy.optimize import minimize

from build_em_seq import build_em_sequences  # импорт из твоего файла

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
):
    """
    EM: на E-шаге считаем gamma, xi для текущих A_t,B_t;
        на M-шаге численно максимизируем Q(w) – регуляризацию.
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
        p_break=p_break_init,   # тут важно
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

        print(f"[EM] it={it}, logL(before M)={logL:.5f}, Q(old)={-negQ(w, gamma, xi):.5f}")

        # M-шаг (численная оптимизация)
        res = minimize(
            fun=lambda w_: negQ(w_, gamma, xi),
            x0=w,
            method="L-BFGS-B",
            options={"maxiter": 60}
        )

        w = res.x
        print(f"         Q(new)={-negQ(w, gamma, xi):.5f}")

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

    return rel_opt, p_look_opt, matrix_click_opt, p_break_opt


def demo(n_em_iters, sostoyanie=1, n_chains=5, n_pairs=20):
    """
    Демо: генерим delivery, берём одну цепочку наблюдений
    и прогоняем EM.
    """
    rel_matrix_full = np.array(P_rel_ref, dtype=float)
    P_look = np.array(P_look_ref, dtype=float)
    matrix_click_full = np.array(P_click_ref, dtype=float)
    p_break_true = float(P_break_ref)

    print(rel_matrix_full)
    print("rel_matrix_full.shape =", rel_matrix_full.shape)

    # (если тебе реально нужен delivery / цвета, оставляем как было)
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
    n_states = 13

    print("rel_matrix_full.shape =", rel_matrix_full.shape)
    print("delivery_full.shape   =", delivery_full.shape)

    list_z, list_o = build_em_sequences(
        n_chains=n_chains,
        n_pairs=n_pairs,
    )

    # Сводка по длинам
    lengths = [len(o) for o in list_o]
    print("--- Сводка по сгенерированным последовательностям ---")
    print("Длины o_seq по цепочкам:", lengths)
    print("Суммарное число наблюдений:", sum(lengths))

    # Берём цепочку с максимальной длиной (или любую с длиной ≥ 2)
    idx_chain = int(np.argmax(lengths))
    if lengths[idx_chain] < 2:
        raise ValueError(
            "Все цепочки слишком короткие (<2 наблюдений) для EM. "
            "Увеличь n_chains или n_pairs."
        )

    o_seq_chain = np.array(list_o[idx_chain], dtype=int)
    print(f"\nБерём цепочку #{idx_chain} для EM")
    print("o_seq этой цепочки:", o_seq_chain)

    # Время в HMM = длина этой цепочки, но не больше доступных строк rel_matrix_full
    T_steps = min(len(o_seq_chain), rel_matrix_full.shape[0])
    obs_seq = o_seq_chain[:T_steps]
    print("Используемый для EM obs_seq (обрезанный до T_steps):", obs_seq)
    print("Длина obs_seq:", len(obs_seq))
    # обрезаем P_rel и P_click под эту длину
    rel_matrix = rel_matrix_full[:T_steps].copy()
    matrix_click = matrix_click_full[:T_steps].copy()
    delivery = delivery_full[:T_steps].copy()


    start_vec = get_start_states(
        rel_matrix[0:2, :],
        P_look,
        delivery[0:2, :],
        p_break=0.05,
    )

    pi = start_vec.ravel().astype(float)
    s = pi.sum()
    if s > 0:
        pi /= s
    else:
        pi = np.full(n_states, 1.0 / n_states)

    print("\nНачальное распределение pi:\n", pi)

    # Дебаг A/B и E-шаг
    A_list = []
    B_list = []

    for i in range(T_steps - 1):
        T_mat = get_transition_matrix(
            rel_matrix[i:i + 2, :],
            P_look,
            delivery[i:i + 2, :],
            p_break=0.05,
        )
        A_list.append(T_mat)

        # ВАЖНО: НЕ переопределяем matrix_click, а берём срез
        click_row = matrix_click[i:i + 1, :]   # shape (1, 2)
        E_mat = get_emission_matrix(click_row, p_break=0.15)
        B_list.append(E_mat)

        if sostoyanie == 1 and i == 0:
            row_sums = T_mat.sum(axis=1)
            print("\nA_list[0] row_sums (min={:.3f}, max={:.3f})"
                  .format(row_sums.min(), row_sums.max()))
            B_row_sums = E_mat.sum(axis=1)
            print("B_list[0] row_sums (min={:.3f}, max={:.3f})"
                  .format(B_row_sums.min(), B_row_sums.max()))

    matrix_click_last = matrix_click[-1:, :]
    E_last = get_emission_matrix(matrix_click_last, p_break=0.15)
    B_list.append(E_last)

    alpha, beta, gamma, xi, logL = e_step(pi, A_list, B_list, obs_seq)

    print("\n=== Результаты E-шагa (с реальными наблюдениями) ===")
    print("log P(O_{1:T} | W) =", logL)
    print("Форма alpha:", alpha.shape)
    print("Форма beta :", beta.shape)
    print("Форма gamma:", gamma.shape)
    print("Форма xi   :", xi.shape)

    print("gamma sum over states at t=0..2:",
          [gamma[t].sum() for t in range(min(3, gamma.shape[0]))])
    print("xi sum over i,j at t=0..2:",
          [xi[t].sum() for t in range(min(3, xi.shape[0]))])

    if sostoyanie == 1:
        print("\nalpha:\n", alpha)
        print("\nbeta:\n", beta)
        print("\ngamma:\n", gamma)
        print("\nxi:\n", xi)

    print("\nGamma[0] (P(z^{(1)} = s_i | O_{1:T})):")
    for s_idx in range(n_states):
        if gamma[0, s_idx] > 1e-3:
            print(f"  P({S_dict[s_idx]} | O_1:T) = {gamma[0, s_idx]:.4f}")
    print("======================================\n")

    rel_opt, p_look_opt, matrix_click_opt, p_break_opt = em_with_tensor_AB(
        rel_matrix_init=rel_matrix,
        p_look_init=P_look,
        matrix_click_init=matrix_click.copy(),  # теперь это настоящий P_click
        delivery=delivery,
        obs_seq=obs_seq,
        pi=pi,
        n_em_iters=n_em_iters,
        p_break_init=p_break_true,             # пробросили P_break из reference
    )


    print("\n===== РЕЗУЛЬТАТЫ EM-ОБУЧЕНИЯ =====")
    np.set_printoptions(precision=3, suppress=True)
    print("Исходная rel_matrix (обрезанная до T_steps):\n", rel_matrix)
    print("Обученная rel_matrix:\n", rel_opt)

    print("\nИсходная p_look:\n", P_look)
    print("Обученная p_look:\n", p_look_opt)

    print("\nИсходная matrix_click (обрезанная):\n", rel_matrix)
    print("Обученная matrix_click:\n", matrix_click_opt)

    print("\np_break_opt =", p_break_opt)


if __name__ == "__main__":
    # для реального запуска можешь раскомментить:
    demo(100, sostoyanie=0, n_chains=100, n_pairs=20)
    