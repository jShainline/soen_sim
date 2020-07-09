interpolation = False
                if interpolation == True:
                    
                    # linear interpolation
                    if I_drive_ind < len(I_drive_list)-1:
                        
                        if I_drive_list[I_drive_ind] > I_drive:
                            
                            I_si_ind_1 = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - I_si_vec[ii])).argmin()
                            I_si_ind_2 = (np.abs(np.asarray(I_si_array[I_drive_ind+1][:]) - I_si_vec[ii])).argmin()
                            
                            slope = rate_array[I_drive_ind+1][I_si_ind_2] - rate_array[I_drive_ind][I_si_ind_1]
                            xx = I_drive_list[I_drive_ind] - I_drive
                            offset = rate_array[I_drive_ind][I_si_ind_1]
                            rate_term = slope*xx+offset
                            
                        else:
                            
                            I_si_ind_1 = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - I_si_vec[ii])).argmin()
                            I_si_ind_2 = (np.abs(np.asarray(I_si_array[I_drive_ind-1][:]) - I_si_vec[ii])).argmin()
                            
                            slope = rate_array[I_drive_ind][I_si_ind_1] - rate_array[I_drive_ind-1][I_si_ind_2]
                            xx = I_drive - I_drive_list[I_drive_ind]
                            offset = rate_array[I_drive_ind-1][I_si_ind_2]
                            rate_term = slope*xx+offset
                            
                        gf = dt*I_fq*rate_term# growth factor
                            
                    else:
                        
                        gf = dt*I_fq*rate_array[I_drive_ind][I_si_ind] # growth factor
                                            
                    
                else: