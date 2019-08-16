module ML_features

    use nn    implicit none
    contains

    subroutine eval_beta(rho,strain_mag,vort_mag,mu,nu_SA,wall_dist,upvp,beta)
        
        implicit none
        
        integer, parameter                                       :: nfeatures = 5
        real*8, dimension(:), intent(in)                         :: rho, strain_mag, vort_mag, mu, nu_SA, wall_dist, upvp
        real*8, dimension(:), intent(out)                        :: beta
        
        real*8, dimension(nfeatures_temp,size(rho)) :: features
        real*8, dimension(6) :: opt_params
        integer :: n_layers, n_weights
        integer, dimension(:), allocatable :: n_neurons        character(len=10) :: act_fn_name
        
        real*8, dimension(size(rho))      :: upvp, chi_SA, fv1_SA, fv2_SA, vort_SA, r_SA, g_SA, fw_SA, production, destruction
        
        chi_SA     = nu_SA*rho/mu
        fv1_SA     = chi_SA**3/(chi_SA**3+357.911D0)
        fv2_SA     = 1.0D0 - chi_SA/(1.0D0 + chi_SA*fv1_SA)
        vort_SA    = vort_mag + fv2_SA * nu_SA/(0.41D0*wall_dist)**2
        r_SA       = nu_SA / vort_SA / (0.41*wall_dist)**2
        g_SA       = 0.3D0*r_SA**6 + 0.7D0*r_SA
        fw_SA      = g_SA * (65.0D0/(g_SA**6 + 64.0D0))**(1.0D0/6.0D0)
        production  = 0.1355D0 * nu_SA * vort_SA
        destruction = (0.1355D0/0.41D0**2 + 2.622D0*1.5D0) * fw_SA * mu_T**2/wall_dist**2/rho**2
        features(1,:) = rho * vort_mag * wall_dist**2 / (mu_T + 1.0)
        features(1,:) = (features(1,:) - dble(1.505232553357033e-05))/dble(5.169869534024708e-05)
        features(2,:) = chi_SA
        features(2,:) = (features(2,:) - dble(4.943931512771335e+02))/dble(1.806691208130323e+03)
        features(3,:) = destruction / (production + 1e-10)
        features(3,:) = (features(3,:) - dble(5.850916223281904e+03))/dble(2.464194220818737e+04)
        features(4,:) = upvp
        features(4,:) = (features(4,:) - dble(2.695030424424496e-04))/dble(5.624035797601320e-03)
        features(5,:) = wall_dist
        features(5,:) = (features(5,:) - dble(1.208007895592732e-01))/dble(2.171529214322337e-01)
        
        open(10, file='nn_config.dat', form='formatted', status='old')
        read(10, *) n_layers
        allocate(n_neurons(n_layers))
        do i=1,n_layers
            read(10, *) n_neurons(i)
        end do
        read(10, *) act_fn_name
        read(10, *) n_weights
        read(10, *) opt_params(1)
        read(10, *) opt_params(2)
        read(10, *) opt_params(3)
        read(10, *) opt_params(4)
        read(10, *) opt_params(5)
        read(10, *) opt_params(6)
        close(10)
        open(20, file='weights.dat', form='formatted', status='old')
        allocate(weights(n_weights))
        do i=1,n_weights
            read(10, *) weights(i)
        end do
        close(20)
        call nn_predict(n_neurons, act_fn_name, 'mse', 'adam', n_weights, weights, nfeatures, size(rho), features, beta, opt_params)
        deallocate(n_neurons)
        deallocate(weights)
    end subroutine eval_features

end module ML_features