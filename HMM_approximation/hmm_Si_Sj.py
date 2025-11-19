'''
### s0-s3 выжил, s0B-s3B купил ушел, s0D-s3D None ушел

s0, s0B, s0D ход предыдущий L -> вниз (L)
s1, s1B, s1D ход предудущий R -> влево, вниз (L)
s2, s2B, s2D ход предыдущий R -> вниз (R)
s3, s3B, s3D ход предыдущий L -> вправо, вниз (R)


s0B, s0D ход предыдущий L -> остаемся в этой клетке - купил ушел/ ушел 
s1B, s1D ход предудущий R -> влево, остались в клетке L - купил ушел/ ушел 
s2B, s2D ход предыдущий R -> остаемся в этой клетке - купил ушел/ ушел
s3B, s3D ход предыдущий L -> влево, остались в клетке R - купил ушел/ ушел

S0 -> S0, S3, S0B, S0D, S3B, S3D
S1 -> S0, S3, S0B, S0D, S3B, S3D
S2 -> S1, S2, S1B, S1D, S2B, S2D
S3 -> S1, S2, S1B, S1D, S2B, S2D
'''



def get_transition_matrix(rel_matrix, p_look_matrix, p_break = 0.15):
    # 0_0 0_1
    # 1_0 1_1
    #"S0", "S1", "S2", "S3", "S0B", "S1B", "S2B", "S3B", "S0D", "S1D", "S2D", "S3D",  "None"
    transition_matrix = np.zeros((13, 13))
    #first row
    p_move_right = p_look_matrix[0, 1] / (p_look_matrix[1, 0] + p_look_matrix[0, 1])
    p_move_left_down = p_look_matrix[1, 0] / (p_look_matrix[1, 0] + p_look_matrix[0, 1])
    transition_matrix[0, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down # P(s1|s1)
    transition_matrix[0, 1] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break) # P(s2|s1)
    transition_matrix[0, 4] = rel_matrix[0, 0] # P(s5|s1)
    transition_matrix[0, 5] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1] # P(s6|s1)
    transition_matrix[0, 8] = (1 - rel_matrix[0, 0]) * p_break # P(s9|s1)
    transition_matrix[0, 9] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break # P(s10|s1)

    #second row
    p_move_left = p_look_matrix[0, 0] / (p_look_matrix[0, 0] + p_look_matrix[1, 1])
    p_move_right_down = p_look_matrix[1, 1] / (p_look_matrix[0, 0] + p_look_matrix[1, 1])
    transition_matrix[1, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down # P(s3|s2)
    transition_matrix[1, 3] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break) # P(s4|s2)
    transition_matrix[1, 6] = rel_matrix[0, 1] # P(s7|s2)
    transition_matrix[1, 7] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0] # P(s8|s2)
    transition_matrix[1, 10] = (1 - rel_matrix[0, 1]) * p_break # P(s11|s2)
    transition_matrix[1, 11] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break # P(s12|s2)

    #third row
    transition_matrix[2, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down # P(s3|s3)
    transition_matrix[2, 3] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break) # P(s4|s3)
    transition_matrix[2, 6] = rel_matrix[0, 1] # P(s7|s3)
    transition_matrix[2, 7] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0] # P(s8|s3)
    transition_matrix[2, 10] = (1 - rel_matrix[0, 1]) * p_break # P(s11|s3)
    transition_matrix[2, 11] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break # P(s12|s3)

    #fourth row
    transition_matrix[3, 0] = (1 - rel_matrix[0,0]) * (1 - p_break) * p_move_left_down # P(s1|s4)
    transition_matrix[3, 1] = (1 - rel_matrix[0,0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0,1]) * (1 - p_break) # P(s2|s4)
    transition_matrix[3, 4] = rel_matrix[0,0] # P(s5|s4)
    transition_matrix[3, 5] = (1 - rel_matrix[0,0]) * (1 - p_break) * p_move_right * rel_matrix[0,1] # P(s6|s4)
    transition_matrix[3, 8] = (1 - rel_matrix[0,0]) * p_break # P(s9|s4)
    transition_matrix[3, 9] = (1 - rel_matrix[0,0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0,1]) * p_break # P(s10|s4)

    #rest_rows
    transition_matrix[4:, 12] = 1.0  # P(s13)

    return transition_matrix