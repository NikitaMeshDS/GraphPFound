'''
S0-S3 и т д из hmm_matrix.py

Она зависит только от P_click!!!!!
N - ничего, C - клик, B - покупка
    R       L
X0  N       N
X1  C       N
X2  N       C
X3  C       C
X4  C,B     N
X5  N       C,B
X6  C,B     C
X7  C       C,B

для генерации будет удобен такой вариант отображения в виде кортежей
    R       L
X0  (0,0)     (0,0)
X1  (1,0)     (0,0)
X2  (0,0)     (1,0)
X3  (1,0)     (1,0)
X4  (1,1)     (0,0)
X5  (0,0)     (1,1)
X6  (1,1)     (1,0)
X7  (1,0)     (1,1)

matrix_click - семпл из нашей выдачи
вида
[
    [0.1, 0.2] i ая полка
]

если рассмотреть пример для S0 -> S3, когда мы начали только, то S0 первое ничего не должно выплюнуть
а вот после перехода когда мы сделали S3 мы можем узнать что было на полке 1

просто условно верояности клика 
'''

import numpy as np
def get_emission_matrix(matrix_click, p_break = 0.15):
    transition_matrix = np.zeros((13, 9))
    pcL = matrix_click[0]
    pcR = matrix_click[1]
    pbcL = matrix_buy_click[0]
    pbcR = matrix_buy_click[1]
    #S0 row
    transition_matrix[0, 0] = (1 - pcL)
    transition_matrix[0, 1] =  pcL 

    #S1 row
    transition_matrix[1, 0] = (1 - pcL) * (1 - pcR)
    transition_matrix[1, 1] = (1 - pcL) * pcR
    transition_matrix[1, 2] = pcL * (1 - pcR)
    transition_matrix[1, 3] = pcL * pcR

    #S2 row
    transition_matrix[2, 0] = (1 - pcR)
    transition_matrix[2, 2] = pcR

    # #S3 row
    transition_matrix[3, 0] = (1 - pcR) * (1 - pcL)
    transition_matrix[3, 1] = pcR * (1 - pcL)
    transition_matrix[3, 2] = (1 - pcR) * pcL
    transition_matrix[3, 3] = pcR * pcL

    #S0D
    transition_matrix[4, 0] = 1

    #S1D
    transition_matrix[5, 0] = (1 - pcL)
    transition_matrix[5, 2] = pcL

    #S2D
    transition_matrix[6, 0] = 1

    #S3D
    transition_matrix[7, 0] = (1 - pcR)
    transition_matrix[7, 1] = pcR

    #S0B row
    transition_matrix[8, 4] = 1

    #S1B row 
    transition_matrix[9, 4] = (1 - pcL)
    transition_matrix[9, 6] = pcL

    #S2B row
    transition_matrix[10, 5] = 1

    #S3B row
    transition_matrix[11, 5] = (1 - pcR)
    transition_matrix[11, 7] = pcR

    #NONE row
    transition_matrix[12, 0] = 1



    return transition_matrix
matrix_click = np.array([0.2, 0.3])
m = get_emission_matrix(matrix_click, p_break= 0.15)
# [sum(x) for x in m]
m # столбцы Sки, строки состояний вектор состояний для каждой Ски это просто m[i состояние, 0:8 иксов]