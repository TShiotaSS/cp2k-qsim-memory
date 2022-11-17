!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2021 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Determine active space Hamiltonian
!> \par History
!>      04.2016 created [JGH]
!> \author JGH
! **************************************************************************************************
MODULE qs_active_space_methods
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set,&
                                              get_atomic_kind
   USE basis_set_types,                 ONLY: &
        allocate_sto_basis_set, create_gto_from_sto_basis, deallocate_gto_basis_set, &
        deallocate_sto_basis_set, gto_basis_set_type, init_orb_basis_set, set_sto_basis_set, &
        srules, sto_basis_set_type
   USE cell_types,                      ONLY: cell_type
   USE cp_blacs_env,                    ONLY: cp_blacs_env_type
   USE scf_control_types,               ONLY: scf_control_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_cholesky,               ONLY: cp_dbcsr_cholesky_decompose,&
                                              cp_dbcsr_cholesky_invert
   USE cp_dbcsr_operations,             ONLY: cp_dbcsr_plus_fm_fm_t,&
                                              cp_dbcsr_sm_fm_multiply,&
                                              dbcsr_allocate_matrix_set,&
                                              dbcsr_deallocate_matrix_set,&
                                              copy_fm_to_dbcsr
   USE cp_files,                        ONLY: close_file,&
                                              open_file
   USE cp_fm_basic_linalg,              ONLY: cp_fm_column_scale,&
                                              cp_fm_gemm, cp_fm_upper_to_full,&
                                              cp_fm_scale_and_add, cp_fm_symm, cp_fm_transpose, &
                                              cp_fm_scale, cp_fm_solve, cp_fm_trace, cp_fm_row_scale, &
                                              cp_fm_column_scale
   USE cp_fm_diag,                      ONLY: cp_fm_syevd
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_pool_types,                ONLY: cp_fm_pool_type,&
                                              fm_pool_create,&
                                              fm_pool_release
   USE cp_fm_types,                     ONLY: &
        cp_fm_create, cp_fm_get_element, cp_fm_get_info, cp_fm_p_type, cp_fm_release, &
        cp_fm_set_all, cp_fm_set_element, cp_fm_to_fm, cp_fm_type, cp_fm_get_diag, &
        cp_fm_set_submatrix, cp_fm_get_submatrix, cp_fm_write_formatted
   USE cp_fm_cholesky,                  ONLY: cp_fm_cholesky_decompose,&
                                              cp_fm_cholesky_invert
   USE cp_fm_vect,                      ONLY: cp_fm_vect_dealloc
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm, copy_fm_to_dbcsr, copy_fm_to_dbcsr_bc
   USE cp_gemm_interface,               ONLY: cp_gemm
   USE cp_array_utils,                  ONLY: cp_1d_r_p_type
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_io_unit,&
                                              cp_logger_type
   USE cp_output_handling,              ONLY: &
        cp_p_file, cp_print_key_finished_output, cp_print_key_should_output, cp_print_key_unit_nr, &
        debug_print_level, high_print_level, low_print_level, medium_print_level, &
        silent_print_level, cp_iterate
   USE cp_para_types,                   ONLY: cp_para_env_type
   USE cp_realspace_grid_cube,          ONLY: cp_pw_to_cube
   USE dbcsr_api,                       ONLY: dbcsr_copy,&
                                              dbcsr_add,&
                                              dbcsr_deallocate_matrix,&
                                              dbcsr_csr_create,&
                                              dbcsr_csr_type,&
                                              dbcsr_p_type,&
                                              dbcsr_type, &
                                              dbcsr_scale, &
                                              dbcsr_set, &
                                              dbcsr_release, &
                                              dbcsr_create, &
                                              dbcsr_print, &
                                              dbcsr_multiply,&
                                              dbcsr_transposed,&
                                              dbcsr_desymmetrize
   USE input_constants,                 ONLY: &
        casci_canonical, dmft_model, eri_method_full_gpw, eri_method_gpw_ht, eri_operator_coulomb, &
        eri_operator_erf, eri_operator_erfc, eri_operator_gaussian, eri_operator_yukawa, hf_model, &
        mao_projection, rsdft_model, wannier_projection, use_mom_ref_zero, use_mom_ref_coac, do_band
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: default_path_length,&
                                              default_string_length,&
                                              dp,&
                                              int_8
   USE mathconstants,                   ONLY: fourpi
   USE memory_utilities,                ONLY: reallocate
   USE message_passing,                 ONLY: mp_bcast
   USE particle_list_types,             ONLY: particle_list_type
   USE particle_types,                  ONLY: particle_type
   USE periodic_table,                  ONLY: ptable
   USE pw_env_methods,                  ONLY: pw_env_create,&
                                              pw_env_rebuild
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_release,&
                                              pw_env_type
   USE pw_grid_types,                   ONLY: pw_grid_type
   USE pw_methods,                      ONLY: pw_integrate_function,&
                                              pw_transfer, pw_zero, pw_axpy, pw_scale
   USE pw_poisson_methods,              ONLY: pw_poisson_rebuild,&
                                              pw_poisson_solve
   USE pw_poisson_types,                ONLY: ANALYTIC0D,&
                                              PERIODIC3D,&
                                              greens_fn_type,&
                                              pw_poisson_analytic,&
                                              pw_poisson_periodic,&
                                              pw_poisson_type
   USE pw_pool_types,                   ONLY: pw_pool_create_pw,&
                                              pw_pool_give_back_pw,&
                                              pw_pool_type
   USE pw_types,                        ONLY: COMPLEXDATA1D,&
                                              REALDATA3D,&
                                              REALSPACE,&
                                              RECIPROCALSPACE,&
                                              pw_p_type,&
                                              pw_release,&
                                              pw_type
   USE qs_active_space_types,           ONLY: active_space_type,&
                                              create_active_space_type,&
                                              csr_idx_from_combined,&
                                              csr_idx_to_combined,&
                                              eri_type,&
                                              eri_type_eri_element_func,&
                                              get_irange_csr,&
                                              release_active_space_type
   USE qs_density_matrices,             ONLY: calculate_density_matrix
   USE qs_active_space_linear,          ONLY: initialize_subspace_linear,&
                                              update_subspace_linear,&
                                              solution_subspace_linear,&
                                              terminate_subspace_linear
   USE qs_collocate_density,            ONLY: calculate_wavefunction, &
                                              calculate_rho_elec
   USE qs_integrate_potential,          ONLY: integrate_v_rspace
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type,&
                                              set_qs_env
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_ks_methods,                   ONLY: qs_ks_update_qs_env
   USE qs_ks_types,                     ONLY: qs_ks_env_type, &
                                              qs_ks_did_change, &
                                              set_ks_env
   USE qs_loc_methods,                  ONLY: qs_loc_driver
   USE qs_loc_types,                    ONLY: qs_loc_env_create,&
                                              qs_loc_env_new_type,&
                                              qs_loc_env_release
   USE qs_loc_utils,                    ONLY: qs_loc_control_init,&
                                              qs_loc_env_init,&
                                              qs_loc_init
   USE qs_mo_methods,                   ONLY: calculate_subspace_eigenvalues,&
                                              make_mo_eig
   USE qs_mo_types,                     ONLY: allocate_mo_set,&
                                              get_mo_set,&
                                              init_mo_set,&
                                              mo_set_p_type,&
                                              mo_set_type,&
                                              deallocate_mo_set
   USE qs_scf_types,                    ONLY: qs_scf_env_type
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE qs_rho_types,                    ONLY: qs_rho_get,&
                                              qs_rho_set,&
                                              qs_rho_type
   USE qs_rho_methods,                  ONLY: qs_rho_update_rho
   USE qs_subsys_types,                 ONLY: qs_subsys_get,&
                                              qs_subsys_type
   USE qs_scf_post_gpw,                 ONLY: make_lumo_gpw
   USE qs_moments,                      ONLY: qs_moment_locop
   USE task_list_methods,               ONLY: generate_qs_task_list
   USE task_list_types,                 ONLY: allocate_task_list,&
                                              deallocate_task_list,&
                                              task_list_type

   USE qs_p_env_types,                  ONLY: qs_p_env_type,&
                                              p_env_release
   USE response_solver,                 ONLY: response_equation,&
                                              ks_ref_potential

   USE qs_force_types,                  ONLY: allocate_qs_force,&
                                              deallocate_qs_force,&
                                              qs_force_type,&
                                              sum_qs_force,&
                                              total_qs_force,&
                                              zero_qs_force,&
                                              replicate_qs_force
   USE lri_environment_types,           ONLY: lri_environment_type
   USE efield_utils,                    ONLY: calculate_ecore_efield
   USE qs_core_hamiltonian,             ONLY: build_core_hamiltonian_matrix
   USE qs_core_energies,                ONLY: calculate_ecore_overlap,&
                                              calculate_ecore_self
   USE ri_environment_methods,          ONLY: build_ri_matrices
   USE qs_external_potential,           ONLY: external_c_potential,&
                                              external_e_potential
   USE qs_overlap,                      ONLY: build_overlap_matrix

#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_active_space_methods'

   PUBLIC :: active_space_main, qs_calculate_forces_active_space

   TYPE, EXTENDS(eri_type_eri_element_func) :: eri_fcidump_print
      INTEGER :: unit_nr, bra_start, ket_start
   CONTAINS
      PROCEDURE :: func => eri_fcidump_print_func
   END TYPE

   TYPE, EXTENDS(eri_type_eri_element_func) :: eri_fcidump_checksum
      INTEGER :: bra_start = 0, ket_start = 0
      REAL(KIND=dp) :: checksum = 0.0_dp
   CONTAINS
      PROCEDURE, PASS :: set => eri_fcidump_set
      PROCEDURE :: func => eri_fcidump_checksum_func
   END TYPE eri_fcidump_checksum

   type rdm_ptr
      type(cp_fm_type), pointer :: ptr
   end type rdm_ptr
! **************************************************************************************************

CONTAINS

! **************************************************************************************************
!> \brief Sets the starting indices of the bra and ket.
!> \param this object reference
!> \param bra_start starting index of the bra
!> \param ket_start starting index of the ket
! **************************************************************************************************
   SUBROUTINE eri_fcidump_set(this, bra_start, ket_start)
      CLASS(eri_fcidump_checksum) :: this
      INTEGER, INTENT(IN) :: bra_start, ket_start
      this%bra_start = bra_start
      this%ket_start = ket_start
   END SUBROUTINE eri_fcidump_set

! **************************************************************************************************
!> \brief Main method for determining the active space Hamiltonian
!> \param input ...
!> \param logger ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE active_space_main(input, logger, qs_env)
      TYPE(section_vals_type), POINTER                   :: input
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(qs_environment_type), POINTER                 :: qs_env

      CHARACTER(len=*), PARAMETER                        :: routineN = 'active_space_main'
      CHARACTER(LEN=default_path_length)                 :: solver_filename, fcidump_filename, &
                                                            output_filename
      CHARACTER(len=10)                                  :: cshell, lnam(5)
      INTEGER :: eri_method, eri_operator, eri_print, handle, i, iatom, ishell, isp, ispin, iw, j, &
         jm, m, mselect, n1, n2, nao, natom, ncol, nel, nelec_active, nelec_inactive, &
         nelec_inactive_alpha, nelec_inactive_beta, nelec_total, nmo, nn1, nn2, nspins, zval, &
         nlumo, nlumos, k, nmo_original, iw2
      INTEGER, DIMENSION(2)                              :: nactive_orb, nepol, ninactive_orb
      INTEGER, DIMENSION(5)                              :: nshell
      INTEGER, DIMENSION(:), POINTER                     :: invals
      INTEGER, DIMENSION(:, :), POINTER                  :: active_orb_list
      LOGICAL                                            :: do_kpoints, ex_operator, ex_perd, &
                                                            explicit, isolated, store_wfn, in_active
      REAL(KIND=dp) :: eri_eps_grid, eri_eps_int, eri_gpw_cutoff, eri_op_param, eri_rcut, &
         eri_rel_cutoff, maxocc, nze_percentage
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: eigenvalues
      TYPE(active_space_type), POINTER                   :: active_space_env
      TYPE(atomic_kind_type), POINTER                    :: atomic_kind
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER          :: mos_localized
      TYPE(cp_fm_type), POINTER                          :: fm_ref, fm_target, mo_coeff, &
                                                            mo_coeff2
      TYPE(cp_para_env_type), POINTER                    :: para_env
      TYPE(dbcsr_csr_type), POINTER                      :: eri_mat
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      TYPE(dbcsr_type), POINTER                          :: denmat, denmat_ref
      TYPE(dbcsr_p_type)                                 :: pmat
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(scf_control_type), POINTER                    :: scf_control
      TYPE(qs_scf_env_type), POINTER                     :: scf_env
      TYPE(gto_basis_set_type), POINTER                  :: pro_basis_set
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos, mo_set_ref
      TYPE(mo_set_type), POINTER                         :: mo_set, mos_extended
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_loc_env_new_type), POINTER                 :: qs_loc_env_occ, qs_loc_env_virt
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(qs_energy_type), POINTER                      :: energy
      TYPE(section_vals_type), POINTER                   :: as_input, loc_print, loc_section, &
                                                            print_orb, lr_section
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER          :: unoccupied_orbs
      TYPE(cp_1d_r_p_type), DIMENSION(:), POINTER        :: unoccupied_evals
      TYPE(cp_fm_pool_type), POINTER                     :: wfn_fm_pool
      TYPE(cp_fm_struct_type), POINTER                   :: wfn_fm_struct, ao_ao_fm_struct, &
                                                            mo_ao_fm_struct, mo_mo_fm_struct, &
                                                            ao_mo_fm_struct, ao_fm_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env

      !for debug
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks, mo_derivs
      TYPE(cp_fm_type), POINTER                          :: matrix_ks_fm, ortho_fm, work_fm, &
                                                            tmp_fm, col_fm, col2_fm

      type(rdm_ptr), DIMENSION(:), allocatable           :: rdm1_fm
      type(rdm_ptr), DIMENSION(:, :, :), allocatable     :: rdm2_fm

      REAL(KIND=dp), DIMENSION(:), POINTER               :: mo_eigenvalues, mo_eigenvalues_original

      REAL(KIND=dp), DIMENSION(:), POINTER               :: ref_point
      TYPE(pw_p_type)                                    :: pot_g, pot_r, rho_tot_g, rho_r, rho_g
      TYPE(pw_env_type), POINTER                         :: pw_env_sub
      TYPE(pw_poisson_type), POINTER                     :: poisson_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(task_list_type), POINTER                      :: task_list_sub
      REAL(KIND=dp)                                      :: progression_factor, cutoff_old, relative_cutoff_old,&
                                                            total_rho, factor
      LOGICAL                                            :: requires_mo_derivs_old
      INTEGER                                            :: i_multigrid, n_multigrid, nkind
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: natom_of_kind
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: e_cutoff_old
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: col1

      ! Z-vector equation
      TYPE(qs_p_env_type), POINTER                       :: p_env
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER          :: zmos, ztrans, kstot_fm
      INTEGER, DIMENSION(2)                              :: naocc, navirt
      TYPE(qs_force_type), POINTER, DIMENSION(:)         :: force

      !--------------------------------------------------------------------------------------------!

      as_input => section_vals_get_subs_vals(input, "DFT%PRINT%ACTIVE_SPACE")
      CALL section_vals_get(as_input, explicit=explicit)
      IF (.NOT. explicit) RETURN
      CALL timeset(routineN, handle)

      iw = cp_logger_get_default_io_unit(logger)

      IF (iw > 0) THEN
         WRITE (iw, '(/,T2,A)') &
            '!-----------------------------------------------------------------------------!'
         WRITE (iw, '(T26,A)') "Generate Active Space Hamiltonian"
         WRITE (iw, '(T27,A)') "Interface to CAS-CI and DMRG-CI"
         WRITE (iw, '(T2,A)') &
            '!-----------------------------------------------------------------------------!'
      END IF

      requires_mo_derivs_old = qs_env%requires_mo_derivs
      qs_env%requires_mo_derivs = .False.

      ! k-points?
      CALL get_qs_env(qs_env, do_kpoints=do_kpoints, energy=energy)
      IF (do_kpoints) THEN
         CALL cp_abort(__LOCATION__, "K-points not allowed for Active Space Interface")
      END IF

      call get_qs_env(qs_env, active_space=active_space_env)
      if (.not. ASSOCIATED(active_space_env)) CALL create_active_space_type(active_space_env)

      active_space_env%energy_total = 0.0_dp
      active_space_env%energy_ref = energy%total
      active_space_env%energy_inactive = 0.0_dp
      active_space_env%energy_active = 0.0_dp

      ! input options

      ! model
      CALL section_vals_val_get(as_input, "MODEL", i_val=active_space_env%model)
      IF (iw > 0) THEN
         SELECT CASE (active_space_env%model)
         CASE (hf_model)
            WRITE (iw, '(T4,A)') "Hartree-Fock model for interaction Hamiltonian"
         CASE (rsdft_model)
            WRITE (iw, '(T4,A)') "Range-separated DFT model for interaction Hamiltonian"
         CASE (dmft_model)
            WRITE (iw, '(T4,A)') "DMFT model for interaction Hamiltonian"
         CASE DEFAULT
            CPABORT("Unknown Model")
         END SELECT
      END IF

      ! isolated (molecular) system?
      CALL section_vals_val_get(as_input, "ISOLATED_SYSTEM", l_val=isolated)
      active_space_env%molecule = isolated
      IF (iw > 0) THEN
         IF (active_space_env%molecule) THEN
            WRITE (iw, '(T4,A)') "System is treated without periodicity"
         END IF
      END IF

      CALL section_vals_val_get(as_input, "ACTIVE_ELECTRONS", i_val=nelec_active)
      IF (nelec_active < 0) CPABORT("Specify a positive number of active electrons.")
      CALL get_qs_env(qs_env, nelectron_total=nelec_total, nelectron_spin=nepol)
      CALL get_qs_env(qs_env, dft_control=dft_control, scf_control=scf_control)
      nspins = dft_control%nspins
      CALL section_vals_val_get(as_input, "INACTIVE_ELECTRONS", explicit=explicit)
      CALL section_vals_val_get(as_input, "INACTIVE_ELECTRONS", i_vals=invals)
      IF (nspins > 1) THEN
         IF (.NOT. explicit) THEN
            CALL cp_abort(__LOCATION__, "Number of Inactive Electrons has to be specified"// &
                          " in spin polarised case.")
         END IF
         nelec_inactive_alpha = invals(1)
         nelec_inactive_beta = invals(2)
         nelec_inactive = nelec_inactive_alpha + nelec_inactive_beta
      ELSE
         IF (explicit) THEN
            nelec_inactive = invals(1)
         ELSE
            nelec_inactive = nelec_total - nelec_active
         END IF
         nelec_inactive_alpha = nelec_inactive
         nelec_inactive_beta = 0
      END IF

      IF (iw > 0) THEN
         IF (nspins < 2) THEN
            WRITE (iw, '(T4,A,T69,I10)') "Total number of electrons", nelec_total
            WRITE (iw, '(T4,A,T69,I10)') "Number of active electrons", nelec_active
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive electrons", nelec_inactive
         ELSE
            WRITE (iw, '(T4,A,T69,I10)') "Total number of electrons", nelec_total
            WRITE (iw, '(T4,A,T69,I10)') "Number of active electrons", nelec_active
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive electrons", nelec_inactive
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive electrons (alpha)", nelec_inactive_alpha
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive electrons (beta)", nelec_inactive_beta
         END IF
      END IF

      CPASSERT(nelec_inactive >= 0)
      CPASSERT(nelec_inactive_alpha >= 0)
      CPASSERT(nelec_inactive_beta >= 0)
      CPASSERT(nelec_total == nelec_inactive + nelec_active)
      IF (nspins > 1) THEN
         CPASSERT(nepol(1) >= nelec_inactive_alpha)
         CPASSERT(nepol(2) >= nelec_inactive_beta)
      END IF

      active_space_env%nelec_active = nelec_active
      active_space_env%nelec_inactive = nelec_inactive
      IF (nspins == 1) THEN
         active_space_env%nelec_total(1) = nelec_total
         active_space_env%nelec_total(2) = 0
      ELSE
         active_space_env%nelec_total(1) = nepol(1)
         active_space_env%nelec_total(2) = nepol(2)
      END IF
      active_space_env%nelec_inactive_spinwise(1) = nelec_inactive_alpha
      active_space_env%nelec_inactive_spinwise(2) = nelec_inactive_beta
      active_space_env%multiplicity = dft_control%multiplicity
      active_space_env%nspins = nspins

      CALL get_qs_env(qs_env, rho=rho)
      CALL qs_rho_get(rho, rho_ao=rho_ao)
      CALL dbcsr_allocate_matrix_set(active_space_env%pmat_reference, nspins)
      DO ispin = 1, nspins
         ALLOCATE (denmat_ref)
         CALL dbcsr_copy(denmat_ref, rho_ao(ispin)%matrix)
         active_space_env%pmat_reference(ispin)%matrix => denmat_ref
      END DO

      ! define the active/inactive space orbitals
      CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS", explicit=explicit)
      CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS", i_vals=invals)
      IF (.NOT. explicit) THEN
         CALL cp_abort(__LOCATION__, "Number of Active Orbitals has to be specified.")
      END IF

      IF (SIZE(invals) /= 1) CPABORT("Expected ONE entry in the ACTIVE_ORBITALS")
      DO ispin = 1, nspins
         nactive_orb(ispin) = invals(1)
      END DO
      ninactive_orb = 0
      CALL section_vals_val_get(as_input, "INACTIVE_ORBITALS", explicit=explicit)
      CALL section_vals_val_get(as_input, "INACTIVE_ORBITALS", i_vals=invals)
      IF (.NOT. explicit) THEN
         IF (nspins == 1) THEN
            ninactive_orb(1) = active_space_env%nelec_inactive_spinwise(1)/2
         ELSE
            DO ispin = 1, nspins
               ninactive_orb(ispin) = active_space_env%nelec_inactive_spinwise(ispin)
            END DO
         END IF
      ELSE
         CPASSERT(SIZE(invals) >= nspins)
         DO ispin = 1, nspins
            ninactive_orb(ispin) = invals(ispin)
         END DO
      END IF

      active_orb_list => active_space_env%active_orb_list

      CALL get_qs_env(qs_env, mos=mos)
      if (.not. ASSOCIATED(active_orb_list)) THEN
        allocate(active_orb_list(max(nactive_orb(1), nactive_orb(2)), nspins))
        active_space_env%active_orb_list => active_orb_list

        CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS_ALIST", explicit=explicit)
        CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS_ALIST", i_vals=invals)
        if (.not. explicit) CPABORT("ACTIVE_ORBITALS_ALIST needs to be specified")

        if (SIZE(invals) .ne. nactive_orb(1)) then
          CPABORT("ACTIVE_ORBITALS does not correpond with length of ACTIVE_ORBITALS_ALIST")
        endif
        CPASSERT(SIZE(invals) == nactive_orb(1))
        do i = 1, SIZE(invals)
          active_orb_list(i, 1) = invals(i)
        enddo
        if (nspins > 1) then
          CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS_BLIST", explicit=explicit)
          CALL section_vals_val_get(as_input, "ACTIVE_ORBITALS_BLIST", i_vals=invals)
          if (explicit) then
            CPASSERT(SIZE(invals) == nactive_orb(2))
            do i = 1, SIZE(invals)
              active_orb_list(i, 2) = invals(i)
            enddo
          endif
        endif
      endif
      do ispin = 1, nspins
        call get_mo_set(mo_set=mos(ispin)%mo_set, nao=nao, nmo=nmo)
        naocc(ispin) = 0
        do i = 1, nactive_orb(ispin)
          m = active_orb_list(i, ispin)
          if (m <= nmo) then
            naocc(ispin) = naocc(ispin) + 1
          endif
        enddo
        navirt(ispin) = nactive_orb(ispin) - naocc(ispin)
      enddo
      nlumo = maxval(active_orb_list) - nmo
      nlumo = min(nlumo+1, nao-nmo)

      ! calculate necessary unoccupied orbitals
      NULLIFY(unoccupied_orbs, unoccupied_evals)
      if (nlumo > 0) then
        ALLOCATE(unoccupied_orbs(nspins))
        ALLOCATE(unoccupied_evals(nspins))
        call get_qs_env(qs_env, scf_env=scf_env)
        call make_lumo_gpw(qs_env, scf_env, unoccupied_orbs, unoccupied_evals, nlumo, nlumos)
        CPASSERT(nlumo == nlumos)
      endif

      IF (nspins == 1) THEN
         CPASSERT(MOD(nelec_inactive, 2) == 0)
         CPASSERT(ninactive_orb(1) >= nelec_inactive/2)
      ELSE
         CPASSERT(ninactive_orb(1) >= nelec_inactive_alpha)
         CPASSERT(ninactive_orb(2) >= nelec_inactive_beta)
      END IF

      CALL get_qs_env(qs_env, mos=mos)
      IF (.NOT. ASSOCIATED(mos(1)%mo_set)) THEN
         CALL cp_abort(__LOCATION__, "A reference MO set has to be available when determining"// &
                       " the active space orbitals.")
      END IF

      ! if moset is dbcsr, set fm
      do ispin = 1, nspins
        if (mos(ispin)%mo_set%use_mo_coeff_b) then
          CALL copy_dbcsr_to_fm(mos(ispin)%mo_set%mo_coeff_b, mos(ispin)%mo_set%mo_coeff)
          mos(ispin)%mo_set%use_mo_coeff_b = .false.
        endif
      enddo

      ! rotate to make orbitals canonical
      call get_qs_env(qs_env, mos=mos, matrix_ks=matrix_ks, mo_derivs=mo_derivs)
      call make_mo_eig(mos, nspins, matrix_ks, scf_control, mo_derivs)

      ! Orbital tracking does not work for band calculations
      IF (qs_env%run_type_id /= do_band .and. ASSOCIATED(active_space_env%mos_active_grid)) THEN
        CALL get_qs_env(qs_env, pw_env=pw_env_sub)
        CALL orbital_overlaps(mos, unoccupied_orbs, active_space_env%mos_active_grid, &
                              qs_env, pw_env_sub, active_orb_list, logger)
      END IF
      do ispin = 1, nspins
        call IntegerQSort(active_orb_list(:nactive_orb(ispin), ispin))
      end do


      ! print out active orbital lists
      if (iw > 0) then
          write(iw, *) "active orbital list:"
          if (nspins > 1) then
              write(iw, *) "alpha"
              write(iw, *) active_orb_list(1:nactive_orb(1), 1)
              write(iw, *) "beta"
              write(iw, *) active_orb_list(1:nactive_orb(2), 2)
          else
              write(iw, *) active_orb_list(1:nactive_orb(1), 1)
          endif
      endif


      call get_qs_env(qs_env, mos=mos, blacs_env=blacs_env)
      allocate(mo_set_ref(nspins))
      do ispin = 1, nspins
        !create an extended mos
        CALL get_mo_set(mo_set=mos(ispin)%mo_set, maxocc=maxocc, nao=nao, nmo=nmo_original,&
                        eigenvalues=mo_eigenvalues_original)
        nmo = nlumo+nmo_original
        nel = active_space_env%nelec_inactive_spinwise(ispin)

        CALL cp_fm_struct_create(wfn_fm_struct, nrow_global=nao, ncol_global=nmo, context=blacs_env)
        CALL fm_pool_create(wfn_fm_pool, wfn_fm_struct)
        nullify(mos_extended)
        CALL allocate_mo_set(mos_extended, nao, nmo, nel, &
                             REAL(nel, dp), maxocc, flexible_electron_count=0.0_dp)
        CALL init_mo_set(mos_extended, fm_pool=wfn_fm_pool, name="mos-extended")
        CALL fm_pool_release(wfn_fm_pool)
        CALL cp_fm_struct_release(wfn_fm_struct)

        call get_mo_set(mo_set=mos(ispin)%mo_set, mo_coeff=mo_coeff)
        call get_mo_set(mo_set=mos_extended, mo_coeff=mo_coeff2, eigenvalues=mo_eigenvalues)

        !fill in eigenvalues
        mo_eigenvalues(1:nmo_original) = mo_eigenvalues_original(1:nmo_original)
        if (nlumo /= 0) then
          mo_eigenvalues(nmo_original+1:nmo) = unoccupied_evals(ispin)%array(1:nmo-nmo_original)
        endif

        !fill in MOs
        j = 1
        do i = 1, nmo
          in_active = .False.
          do k = 1, nactive_orb(ispin)
            if (active_orb_list(k, ispin) == i) then
              in_active = .True.
            endif
          enddo
          if (.not. in_active) then
            if (i <= nmo_original) then
              call cp_fm_to_fm(mo_coeff, mo_coeff2, 1, i, j)
            else
              CPASSERT(i - nmo_original > 0)
              CPASSERT(i - nmo_original <= nlumos)
              call cp_fm_to_fm(unoccupied_orbs(ispin)%matrix, mo_coeff2, &
                               1, i - nmo_original, j)
            endif
            j = j + 1
          endif
          if (j > ninactive_orb(ispin)) then
            exit
          endif
        enddo
        do i = 1, nactive_orb(ispin)
          k = i + ninactive_orb(ispin)
          j = active_orb_list(i, ispin)
          if (j <= nmo_original) then
            call cp_fm_to_fm(mo_coeff, mo_coeff2, 1, j, k)
          else
            CPASSERT(j - nmo_original > 0)
            CPASSERT(j - nmo_original <= nlumos)
            call cp_fm_to_fm(unoccupied_orbs(ispin)%matrix, mo_coeff2, &
                             1, j - nmo_original, k)
          endif
        enddo
        mo_set_ref(ispin)%mo_set => mos(ispin)%mo_set
        mos(ispin)%mo_set => mos_extended
      enddo

      if (associated(unoccupied_evals)) then
        do ispin = 1, nspins
          deallocate(unoccupied_evals(ispin)%array)
        enddo
        deallocate(unoccupied_evals)
      endif


      if (associated(unoccupied_orbs)) then
        do ispin = 1, nspins
          call cp_fm_release(unoccupied_orbs(ispin)%matrix)
        enddo
        deallocate(unoccupied_orbs)
      endif

      CALL get_mo_set(mos(1)%mo_set, mo_coeff=fm_ref, nao=nao)
      maxocc = 2.0_dp
      IF (nspins > 1) maxocc = 1.0_dp
      ALLOCATE (active_space_env%mos_active(nspins))
      ALLOCATE (active_space_env%mos_inactive(nspins))
      DO ispin = 1, nspins
         nmo = nactive_orb(ispin)
         NULLIFY (mo_set)
         CALL allocate_mo_set(mo_set, nao, nmo, 0, 0.0_dp, maxocc, 0.0_dp)
         CALL init_mo_set(mo_set, fm_ref=fm_ref, name="Active Space MO")
         active_space_env%mos_active(ispin)%mo_set => mo_set
         mo_set => active_space_env%mos_inactive(ispin)%mo_set
         nmo = ninactive_orb(ispin)
         nel = active_space_env%nelec_inactive_spinwise(ispin)
         NULLIFY (mo_set)
         CALL allocate_mo_set(mo_set, nao, nmo, nel, REAL(nel, KIND=dp), maxocc, 0.0_dp)
         CALL init_mo_set(mo_set, fm_ref=fm_ref, name="Inactive Space MO")
         active_space_env%mos_inactive(ispin)%mo_set => mo_set
      END DO

      CALL section_vals_val_get(as_input, "ORBITAL_SELECTION", i_val=mselect)
      IF (iw > 0) THEN
         SELECT CASE (mselect)
         CASE DEFAULT
            CPABORT("Unknown orbital selection method")
         CASE (casci_canonical)
            WRITE (iw, '(/,T4,A)') &
               "Active space orbitals selected using energy ordered canonical orbitals"
         CASE (wannier_projection)
            WRITE (iw, '(/,T4,A)') &
               "Active space orbitals selected using projected Wannier orbitals"
         CASE (mao_projection)
            WRITE (iw, '(/,T4,A)') &
               "Active space orbitals selected using modified atomic orbitals (MAO)"
         END SELECT
         IF (nspins < 2) THEN
            WRITE (iw, '(T4,A,T69,I10)') "Number of active orbitals", nactive_orb(1)
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive orbitals", ninactive_orb(1)
         ELSE
            WRITE (iw, '(T4,A,T69,I10)') "Number of active orbitals", SUM(nactive_orb)
            WRITE (iw, '(T4,A,T69,I10)') "Number of active orbitals (alpha)", nactive_orb(1)
            WRITE (iw, '(T4,A,T69,I10)') "Number of active orbitals (beta)", nactive_orb(2)
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive orbitals", SUM(ninactive_orb)
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive orbitals (alpha)", ninactive_orb(1)
            WRITE (iw, '(T4,A,T69,I10)') "Number of inactive orbitals (beta)", ninactive_orb(2)
         END IF
      END IF

      ! get projection spaces
      CALL section_vals_val_get(as_input, "SUBSPACE_ATOM", i_val=iatom, explicit=explicit)
      IF (explicit) THEN
         CALL get_qs_env(qs_env, natom=natom)
         IF (iatom <= 0 .OR. iatom > natom) THEN
            IF (iw > 0) THEN
               WRITE (iw, '(/,T4,A,I3)') "ERROR: SUBSPACE_ATOM number is not valid", iatom
            END IF
            CPABORT("Select a valid SUBSPACE_ATOM")
         END IF
      END IF
      CALL section_vals_val_get(as_input, "SUBSPACE_SHELL", c_val=cshell, explicit=explicit)
      nshell = 0
      lnam = ""
      IF (explicit) THEN
         cshell = ADJUSTL(cshell)
         n1 = 1
         DO i = 1, 5
            ishell = i
            IF (cshell(n1:n1) == " ") THEN
               ishell = ishell - 1
               EXIT
            END IF
            READ (cshell(n1:), "(I1,A1)") nshell(i), lnam(i)
            n1 = n1 + 2
         END DO
      END IF

      ! generate orbitals
      SELECT CASE (mselect)
      CASE DEFAULT
         CPABORT("Unknown orbital selection method")
      CASE (casci_canonical)
         ! create canonical orbitals
         IF (dft_control%restricted) THEN
            CPABORT(" Unclear how we define MOs in the restricted case ... stopping")
         ELSE
            IF (dft_control%do_admm) THEN
               CPABORT("ADMM currently not possible for canonical orbital options")
            END IF
            nmo = 0
            DO ispin = 1, nspins
               CALL get_mo_set(mos(ispin)%mo_set, nmo=m)
               IF (m < ninactive_orb(ispin) + nactive_orb(ispin)) THEN
                  CPABORT("Not enough canonical orbitals available.")
               END IF
               nmo = MAX(m, nmo)
            END DO
            ALLOCATE (eigenvalues(nmo_original+nlumo, nspins))
            IF (mos(1)%mo_set%use_mo_coeff_b) THEN
               CPABORT("Needs MOs in FM format, missing code for DBCSR")
            ELSE
               DO ispin = 1, nspins
                  CALL get_mo_set(mos(ispin)%mo_set, mo_coeff=fm_ref, nmo=nmo, eigenvalues=mo_eigenvalues)
                  eigenvalues(:, ispin) = mo_eigenvalues(:)
                  ! copy inactive orbitals
                  mo_set => active_space_env%mos_inactive(ispin)%mo_set
                  CALL get_mo_set(mo_set, mo_coeff=fm_target)
                  ncol = ninactive_orb(ispin)
                  CALL cp_fm_to_fm(fm_ref, fm_target, ncol, 1, 1)
                  ! copy active orbitals
                  mo_set => active_space_env%mos_active(ispin)%mo_set
                  CALL get_mo_set(mo_set, mo_coeff=fm_target)
                  ncol = nactive_orb(ispin)
                  CALL cp_fm_to_fm(fm_ref, fm_target, ncol, ninactive_orb(ispin) + 1, 1)
               END DO
            END IF
            IF (iw > 0) THEN
               DO ispin = 1, nspins
                  WRITE (iw, '(/,T4,A,I3,T65,A)') "Canonical Orbital Selection for spin", ispin, &
                     "[atomic units]"
                  DO i = 1, SIZE(eigenvalues(:, ispin)), 4
                     jm = MIN(3, SIZE(eigenvalues(:, ispin)) - i)
                     DO j = 0, jm
                       IF (any(active_orb_list(:, ispin) == (i + j))) THEN
                         WRITE (iw, '(T3,(F14.6,A5))', advance="no") eigenvalues(i + j, ispin), " [A]"
                       ELSE
                         WRITE (iw, '(T3,(F14.6,A5))', advance="no") eigenvalues(i + j, ispin), " [I]"
                       ENDIF
                    END DO
                    write (iw, *) ""
                  END DO
               END DO
            END IF
         END IF

      CASE (wannier_projection)
         NULLIFY (loc_section, loc_print)
         loc_section => section_vals_get_subs_vals(as_input, "LOCALIZE")
         CPASSERT(ASSOCIATED(loc_section))
         loc_print => section_vals_get_subs_vals(as_input, "LOCALIZE%PRINT")
         NULLIFY (qs_loc_env_occ, qs_loc_env_virt)
         CALL qs_loc_env_create(qs_loc_env_occ)
         ALLOCATE (mos_localized(nspins))
         DO ispin = 1, nspins
            CALL get_mo_set(mo_set=mos(ispin)%mo_set, mo_coeff=mo_coeff)
            CALL cp_fm_create(mos_localized(ispin)%matrix, mo_coeff%matrix_struct)
            CALL cp_fm_to_fm(mo_coeff, mos_localized(ispin)%matrix)
         END DO
         ! occupied states
         NULLIFY (qs_loc_env_occ)
         CALL qs_loc_env_create(qs_loc_env_occ)
         CALL qs_loc_control_init(qs_loc_env_occ, loc_section, do_homo=.TRUE.)
         CALL qs_loc_init(qs_env, qs_loc_env_occ, loc_section, &
                          mos_localized=mos_localized, do_homo=.TRUE.)
         DO ispin = 1, nspins
            CALL qs_loc_driver(qs_env, qs_loc_env_occ, loc_print, myspin=ispin, &
                               ext_mo_coeff=mos_localized(ispin)%matrix)
         END DO
         ! virtual states
         NULLIFY (qs_loc_env_virt)
         CALL qs_loc_env_create(qs_loc_env_virt)
         CALL qs_loc_control_init(qs_loc_env_virt, loc_section, do_homo=.FALSE.)
         CALL qs_loc_init(qs_env, qs_loc_env_virt, loc_section, &
                          mos_localized=mos_localized, do_homo=.FALSE.)
         CALL qs_loc_env_init(qs_loc_env_virt, qs_loc_env_virt%localized_wfn_control, qs_env, &
                              loc_coeff=mos_localized)
         DO ispin = 1, nspins
            CALL qs_loc_driver(qs_env, qs_loc_env_virt, loc_print, myspin=ispin, &
                               ext_mo_coeff=mos_localized(ispin)%matrix)
         END DO
         !
         ! get definition of subspace
         ! iatom, ishell, nshell, lnam
         CPASSERT(iatom > 0)
         CPASSERT(ishell > 0)
         CALL get_qs_env(qs_env, particle_set=particle_set)
         atomic_kind => particle_set(iatom)%atomic_kind
         CALL get_atomic_kind(atomic_kind=atomic_kind, z=zval)
         NULLIFY (pro_basis_set)
         CALL create_pro_basis(pro_basis_set, zval, ishell, nshell, lnam)
         !
         CALL deallocate_gto_basis_set(pro_basis_set)
         !
         CPABORT("not yet available")
         !
         CALL qs_loc_env_release(qs_loc_env_occ)
         CALL qs_loc_env_release(qs_loc_env_virt)
         CALL cp_fm_vect_dealloc(mos_localized)
         !
         CPABORT("not yet available")
         !
      CASE (mao_projection)
         !
         CPABORT("not yet available")
         !
      END SELECT

      IF (ASSOCIATED(active_space_env%mos_active_grid)) THEN
        DO isp= 1, SIZE(active_space_env%mos_active)
           CALL get_mo_set(mo_set=active_space_env%mos_active(isp)%mo_set, nmo=nmo)
           DO i = 1, nmo
              CALL pw_release(active_space_env%mos_active_grid(i, isp)%pw)
           END DO
        END DO
        deallocate(active_space_env%mos_active_grid)
      END IF

      call get_qs_env(qs_env, pw_env=pw_env_sub)
      call put_orbitals_on_grid(active_space_env%mos_active, &
        active_space_env%mos_active_grid, qs_env, pw_env_sub)

      ! Print orbitals on Cube files
      print_orb => section_vals_get_subs_vals(as_input, "PRINT_ORBITAL_CUBES")
      CALL section_vals_get(print_orb, explicit=explicit)
      IF (explicit) THEN
         !
         CALL print_orbital_cubes(print_orb, qs_env, active_space_env%mos_active)
         !
      END IF

      ! put back the original mos in qs_env, and deallocate the extended
      do ispin = 1, nspins
        call deallocate_mo_set(mos(ispin)%mo_set)
        mos(ispin)%mo_set => mo_set_ref(ispin)%mo_set
      enddo
      deallocate(mo_set_ref)

      ! calculate inactive density matrix
      CALL get_qs_env(qs_env, rho=rho)
      CALL qs_rho_get(rho, rho_ao=rho_ao)
      CPASSERT(ASSOCIATED(rho_ao))
      CALL dbcsr_allocate_matrix_set(active_space_env%pmat_inactive, nspins)
      DO ispin = 1, nspins
         ALLOCATE (denmat)
         CALL dbcsr_copy(denmat, rho_ao(ispin)%matrix)
         mo_set => active_space_env%mos_inactive(ispin)%mo_set
         CALL calculate_density_matrix(mo_set, denmat)
         active_space_env%pmat_inactive(ispin)%matrix => denmat
      END DO

      ! generate integrals
      ! make sure that defaults are set correctly (from basic calculation)
      ! make sure that periodicity is consistent
      CALL section_vals_val_get(as_input, "ERI%METHOD", i_val=eri_method)
      active_space_env%eri%method = eri_method
      CALL section_vals_val_get(as_input, "ERI%OPERATOR", i_val=eri_operator, explicit=ex_operator)
      active_space_env%eri%operator = eri_operator
      CALL section_vals_val_get(as_input, "ERI%OPERATOR_PARAMETER", r_val=eri_op_param)
      active_space_env%eri%operator_parameter = eri_op_param
      CALL section_vals_val_get(as_input, "ERI%CUTOFF_RADIUS", r_val=eri_rcut)
      active_space_env%eri%cutoff_radius = eri_rcut
      CALL section_vals_val_get(as_input, "ERI%PERIODICITY", i_vals=invals, explicit=ex_perd)
      CALL section_vals_val_get(as_input, "ERI%EPS_INTEGRAL", r_val=eri_eps_int)
      active_space_env%eri%eps_integral = eri_eps_int
      IF (active_space_env%molecule) THEN
         ! check that we are in a non-periodic setting
         CALL get_qs_env(qs_env, cell=cell)
         IF (SUM(cell%perd) /= 0) THEN
            CPABORT("Active space option ISOLATED_SYSTEM requires non-periodic setting")
         END IF
         IF (ex_perd) THEN
            IF (SUM(invals) /= 0) THEN
               CPABORT("Active space option ISOLATED_SYSTEM requires non-periodic setting")
            END IF
         END IF
         active_space_env%eri%periodicity(1:3) = 0
      ELSE IF (ex_perd) THEN
         IF (SIZE(invals) == 1) THEN
            active_space_env%eri%periodicity(1:3) = invals(1)
         ELSE
            active_space_env%eri%periodicity(1:3) = invals(1:3)
         END IF
      ELSE
         CALL get_qs_env(qs_env, cell=cell)
         active_space_env%eri%periodicity(1:3) = cell%perd(1:3)
      END IF
      IF (iw > 0) THEN
         WRITE (iw, '(/,T4,A)') "Calculation or Electron Repulsion Intergrals"
         SELECT CASE (eri_method)
         CASE (eri_method_full_gpw)
            WRITE (iw, '(T4,A,T48,A)') "Integration method", "GPW Fourier transform over MOs"
         CASE (eri_method_gpw_ht)
            WRITE (iw, '(T4,A,T43,A)') "Integration method", "Half transformed integrals from GPW"
         CASE DEFAULT
            CPABORT("Unknown ERI method")
         END SELECT
         SELECT CASE (eri_operator)
         CASE (eri_operator_coulomb)
            WRITE (iw, '(T4,A,T40,A,T73,A)') "ERI operator", "Coulomb", "<1/R>"
         CASE (eri_operator_yukawa)
            WRITE (iw, '(T4,A,T40,A,T67,A)') "ERI operator", "Yukawa", "<EXP(-a*R)/R>"
            WRITE (iw, '(T4,A,T63,F14.6)') "ERI operator parameter", eri_op_param
         CASE (eri_operator_erf)
            WRITE (iw, '(T4,A,T40,A,T67,A)') "ERI operator", "Error function", "<ERF(a*R)/R>"
            WRITE (iw, '(T4,A,T63,F14.6)') "ERI operator parameter", eri_op_param
         CASE (eri_operator_erfc)
            WRITE (iw, '(T4,A,T40,A,T66,A)') "ERI operator", "Compl. error function", "<ERFC(a*R)/R>"
            WRITE (iw, '(T4,A,T63,F14.6)') "ERI operator parameter", eri_op_param
         CASE (eri_operator_gaussian)
            WRITE (iw, '(T4,A,T40,A,T66,A)') "ERI operator", "Gaussian attenuated", "<EXP(-a*R^2)/R>"
            WRITE (iw, '(T4,A,T63,F14.6)') "ERI operator parameter", eri_op_param
         CASE DEFAULT
            CPABORT("Unknown ERI operator")
         END SELECT
         WRITE (iw, '(T4,A,T66,E12.4)') "Accuracy of ERI", eri_eps_int
         WRITE (iw, '(T4,A,T69,3I3)') "Periodicity", active_space_env%eri%periodicity(1:3)
         IF (PRODUCT(active_space_env%eri%periodicity(1:3)) == 0) THEN
            IF (eri_rcut > 0.0_dp) WRITE (iw, '(T4,A,T63,F14.6)') "Periodicity (Cutoff)", eri_rcut
         END IF
         IF (nspins < 2) THEN
            WRITE (iw, '(T4,A,T66,I12)') "Total Number of ERI", (nactive_orb(1)**4)/8
         ELSE
            WRITE (iw, '(T4,A,T66,I12)') "Total Number of ERI (aa|aa)", (nactive_orb(1)**4)/8
            WRITE (iw, '(T4,A,T66,I12)') "Total Number of ERI (bb|bb)", (nactive_orb(2)**4)/8
            WRITE (iw, '(T4,A,T66,I12)') "Total Number of ERI (aa|bb)", &
               (nactive_orb(1)**2)*(nactive_orb(2)**2)/4
         END IF
      END IF

      ! allocate container for integrals (CSR matrix)
      CALL get_qs_env(qs_env, para_env=para_env)
      m = (nspins*(nspins + 1))/2
      ALLOCATE (active_space_env%eri%eri(m))
      DO i = 1, m
         ALLOCATE (active_space_env%eri%eri(i)%csr_mat)
         eri_mat => active_space_env%eri%eri(i)%csr_mat
         IF (i == 1) THEN
            n1 = nactive_orb(1)
            n2 = nactive_orb(1)
         ELSEIF (i == 2) THEN
            n1 = nactive_orb(1)
            n2 = nactive_orb(2)
         ELSE
            n1 = nactive_orb(2)
            n2 = nactive_orb(2)
         END IF
         nn1 = (n1*(n1 + 1))/2
         nn2 = (n2*(n2 + 1))/2
         CALL dbcsr_csr_create(eri_mat, nn1, nn2, 0_int_8, 0, 0, para_env%group)
         CALL get_mo_set(active_space_env%mos_active(1)%mo_set, nmo=active_space_env%eri%norb)! here replace i with 1, i can be > nspins
      END DO

      CPASSERT(eri_method == eri_method_full_gpw)

      CALL section_vals_val_get(as_input, "ERI_GPW%EPS_GRID", r_val=eri_eps_grid)
      active_space_env%eri%eri_gpw%eps_grid = eri_eps_grid
      CALL section_vals_val_get(as_input, "ERI_GPW%CUTOFF", r_val=eri_gpw_cutoff)
      active_space_env%eri%eri_gpw%cutoff = eri_gpw_cutoff
      CALL section_vals_val_get(as_input, "ERI_GPW%REL_CUTOFF", r_val=eri_rel_cutoff)
      active_space_env%eri%eri_gpw%rel_cutoff = eri_rel_cutoff
      CALL section_vals_val_get(as_input, "ERI_GPW%PRINT_LEVEL", i_val=eri_print)
      active_space_env%eri%eri_gpw%print_level = eri_print
      CALL section_vals_val_get(as_input, "ERI_GPW%STORE_WFN", l_val=store_wfn)
      active_space_env%eri%eri_gpw%store_wfn = store_wfn
      active_space_env%eri%eri_gpw%redo_poisson = (ex_operator .OR. ex_perd)
      IF (iw > 0) THEN
         WRITE (iw, '(T4,A,T68,F10.4)') "ERI_GPW| Energy cutoff [Ry]", eri_gpw_cutoff
         WRITE (iw, '(T4,A,T68,F10.4)') "ERI_GPW| Relative energy cutoff [Ry]", eri_rel_cutoff
      END IF

      ! creating a new PW environment in pw_env_sub. dft_control is being updated; needs to be reverted
      ! before being used with the original pw_env.
      CALL get_qs_env(qs_env, dft_control=dft_control)
      progression_factor = dft_control%qs_control%progression_factor
      n_multigrid = SIZE(dft_control%qs_control%e_cutoff)
      ALLOCATE (e_cutoff_old(n_multigrid))
      e_cutoff_old(:) = dft_control%qs_control%e_cutoff
      cutoff_old = dft_control%qs_control%cutoff

      dft_control%qs_control%cutoff = active_space_env%eri%eri_gpw%cutoff*0.5_dp
      dft_control%qs_control%e_cutoff(1) = dft_control%qs_control%cutoff
      DO i_multigrid = 2, n_multigrid
         dft_control%qs_control%e_cutoff(i_multigrid) = dft_control%qs_control%e_cutoff(i_multigrid - 1) &
                                                        /progression_factor
      END DO
      relative_cutoff_old = dft_control%qs_control%relative_cutoff
      dft_control%qs_control%relative_cutoff = active_space_env%eri%eri_gpw%rel_cutoff*0.5_dp

      NULLIFY (pw_env_sub)
      CALL pw_env_create(pw_env_sub)
      CALL pw_env_rebuild(pw_env_sub, qs_env)

      ! now calculate ERIs
      CALL calculate_eri_gpw(active_space_env%mos_active, active_space_env%eri, qs_env, iw, pw_env_sub)

      IF (iw > 0) THEN
         DO isp = 1, SIZE(active_space_env%eri%eri)
            eri_mat => active_space_env%eri%eri(isp)%csr_mat
            nze_percentage = 100.0_dp*(REAL(eri_mat%nze_total, KIND=dp) &
                                       /REAL(eri_mat%nrows_total, KIND=dp))/REAL(eri_mat%ncols_total, KIND=dp)
            WRITE (iw, '(T4,A,I2,T30,A,T66,I12)') "ERI_GPW| Spinmatrix:", isp, &
               "Number of  CSR non-zero elements:", eri_mat%nze_total
            WRITE (iw, '(T4,A,I2,T30,A,T66,F12.4)') "ERI_GPW| Spinmatrix:", isp, &
               "Percentage CSR non-zero elements:", nze_percentage
         END DO
      END IF

      ! set the reference active space density matrix
      nspins = active_space_env%nspins
      SELECT CASE (mselect)
      CASE DEFAULT
         CPABORT("Unknown orbital selection method")
      CASE (casci_canonical)
         ! nothing happens
      CASE (wannier_projection)
         CPABORT("NOT IMPLEMENTED")
      CASE (mao_projection)
         CPABORT("NOT IMPLEMENTED")
      END SELECT

      ! calculate one-electron operators in the subspace
      CALL update_active_space(qs_env, active_space_env)

      ! Output a FCIDUMP file
      CALL fcidump(active_space_env, as_input, fcidump_filename)

      ! Get the output FCIDUMP filename
      logger => cp_get_default_logger()
      call cp_iterate(logger%iter_info)
      if (BTEST(cp_print_key_should_output(logger%iter_info, &
                                           as_input, "FCIDUMP"), cp_p_file)) then
          iw2 = cp_print_key_unit_nr(logger, as_input, "FCIDUMP", &
                                     extension=".fcidump", file_status="REPLACE", file_action="READ", file_form="FORMATTED")
          if (para_env%ionode) then
              inquire(unit=iw2, name=output_filename)
          endif
      else
        CPABORT("Failed to create an output file")
      endif

      ! Get the path of the active space executable
      call section_vals_val_get(as_input, "ACTIVE_SPACE_SOLVER", explicit=explicit)
      if (.not.explicit) then
         CPABORT("Active space solver not specified in the input.")
      endif
      CALL section_vals_val_get(as_input, "ACTIVE_SPACE_SOLVER", c_val=solver_filename)

      ! Run active-space solver: This is only called on the IO node
      if (para_env%ionode) then
          call execute_command_line("bash " // trim(solver_filename) // " " // trim(fcidump_filename) &
                                            // " " // trim(output_filename))
      endif
      ! matrices for RDM outputs
      nmo = active_space_env%eri%norb
      allocate(rdm1_fm(nspins))
      allocate(rdm2_fm(nspins, nmo, nmo))

      call read_fcidump(active_space_env, para_env, blacs_env, iw2, iw, rdm1_fm, rdm2_fm)
      call cp_print_key_finished_output(iw2, logger, as_input, "FCIDUMP")

      CALL get_qs_env(qs_env, energy=energy)
      energy%total = active_space_env%energy_total

      CALL cp_fm_struct_create(mo_ao_fm_struct, nrow_global=nmo, ncol_global=nao, context=blacs_env)
      CALL cp_fm_struct_create(ao_mo_fm_struct, nrow_global=nao, ncol_global=nmo, context=blacs_env)
      CALL cp_fm_struct_create(ao_ao_fm_struct, nrow_global=nao, ncol_global=nao, context=blacs_env)
      CALL cp_fm_struct_create(ao_fm_struct, nrow_global=nao, ncol_global=2, context=blacs_env)

      call cp_fm_create(matrix_ks_fm, ao_ao_fm_struct)
      call cp_fm_create(tmp_fm, ao_ao_fm_struct)
      call cp_fm_create(work_fm, ao_mo_fm_struct)
      call cp_fm_create(col_fm, ao_fm_struct)
      call cp_fm_create(col2_fm, ao_fm_struct)
      CALL get_qs_env(qs_env, matrix_ks=matrix_ks)

      ! Calculate the Hartree potential
      CALL pw_env_get(pw_env_sub, auxbas_pw_pool=auxbas_pw_pool, poisson_env=poisson_env)
      NULLIFY (rho_tot_g%pw, pot_r%pw, pot_g%pw)
      CALL pw_pool_create_pw(auxbas_pw_pool, pot_r%pw, use_data=REALDATA3D, in_space=REALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, pot_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_r%pw, use_data=REALDATA3D, in_space=REALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_tot_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)

      allocate(col1(nao, 2))

      CALL get_qs_env(qs_env, rho=rho, ks_env=ks_env)

      ! we need task list
      NULLIFY (task_list_sub)
      CALL allocate_task_list(task_list_sub)
      CALL generate_qs_task_list(ks_env, task_list_sub, &
                                 reorder_rs_grid_ranks=.TRUE., soft_valid=.FALSE., &
                                 skip_load_balance_distributed=dft_control%qs_control%skip_load_balance_distributed, &
                                 pw_env_external=pw_env_sub)

      allocate(ztrans(nspins))
      call get_qs_env(qs_env, atomic_kind_set=atomic_kind_set, force=force)
      nkind = SIZE(atomic_kind_set)
      allocate(natom_of_kind(nkind))
      call get_atomic_kind_set(atomic_kind_set=atomic_kind_set, natom_of_kind=natom_of_kind)
      call allocate_qs_force(active_space_env%rdm2_force, natom_of_kind)
      call zero_qs_force(active_space_env%rdm2_force)
      deallocate(natom_of_kind)
      call set_qs_env(qs_env, force=active_space_env%rdm2_force)

      do ispin = 1, nspins
        call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
        call cp_fm_create(ztrans(ispin)%matrix, mo_coeff%matrix_struct)

        CALL copy_dbcsr_to_fm(matrix_ks(ispin)%matrix, matrix_ks_fm)
        call cp_fm_symm('L', 'U', nao, nmo, 1.0_dp, matrix_ks_fm, mo_coeff, 0.0_dp, work_fm)
        call cp_fm_gemm('N', 'N', nao, nmo, nmo, 1.0_dp, work_fm, rdm1_fm(ispin)%ptr, 0.0_dp, ztrans(ispin)%matrix)

        allocate(pmat%matrix)
        call dbcsr_copy(pmat%matrix, rho_ao(ispin)%matrix)

        do i = 1, nmo
          call cp_fm_get_submatrix(mo_coeff, col1, 1, i, nao, 1)
          call cp_fm_set_submatrix(col_fm, col1, 1, 1, nao, 1)
          do j = i, nmo
            if (i /= j) then
              m = 2
              call cp_fm_get_submatrix(mo_coeff, col1, 1, j, nao, 1)
              call cp_fm_set_submatrix(col_fm, col1, 1, 2, nao, 1)
            else
              m = 1
            endif

            ! denmat is symmetric
            CALL pw_zero(rho_tot_g%pw)
            CALL dbcsr_set(rho_ao(ispin)%matrix, 0.0_dp)
            CALL copy_fm_to_dbcsr(rdm2_fm(ispin, i, j)%ptr, rho_ao(ispin)%matrix)

            CALL pw_zero(rho_r%pw)
            CALL pw_zero(rho_g%pw)
            CALL calculate_rho_elec(matrix_p=rho_ao(ispin)%matrix, rho=rho_r, rho_gspace=rho_g, &
              total_rho=total_rho, pw_env_external=pw_env_sub, ks_env=ks_env, &
              task_list_external=task_list_sub)
            CALL pw_axpy(rho_g%pw, rho_tot_g%pw)

            CALL pw_poisson_solve(poisson_env, rho_tot_g%pw, vhartree=pot_g%pw)
            CALL pw_transfer(pot_g%pw, pot_r%pw)
            CALL pw_scale(pot_r%pw, pot_r%pw%pw_grid%dvol)

            call dbcsr_set(pmat%matrix, 0.0_dp)
            if (i == j) then
              call cp_dbcsr_plus_fm_fm_t(pmat%matrix, col_fm, col_fm, ncol=1)
            else
              call cp_fm_to_fm(mo_coeff, col2_fm, 1, j, 1)
              call cp_dbcsr_plus_fm_fm_t(pmat%matrix, col_fm, col2_fm, ncol=1)
              call cp_dbcsr_plus_fm_fm_t(pmat%matrix, col2_fm, col_fm, ncol=1)
            endif

            CALL dbcsr_set(matrix_ks(ispin)%matrix, 0.0_dp)
            CALL integrate_v_rspace(pot_r, pmat=pmat, &
                                    hmat=matrix_ks(ispin), qs_env=qs_env, &
                                    calculate_forces=.true., pw_env_external=pw_env_sub, &
                                    task_list_external=task_list_sub)
            CALL copy_dbcsr_to_fm(matrix_ks(ispin)%matrix, matrix_ks_fm)
            call cp_fm_symm('L', 'U', nao, m, 1.0_dp, matrix_ks_fm, col_fm, 0.0_dp, col2_fm)

            ! add i-th column of work_fm to j-th column of ztrans
            call cp_fm_get_submatrix(col2_fm, col1, 1, 1, nao, m)
            call cp_fm_set_submatrix(ztrans(ispin)%matrix, col1(:, 1:1), 1, j, nao, 1, 1.0_dp, 1.0_dp)
            if (i /= j) then
              call cp_fm_set_submatrix(ztrans(ispin)%matrix, col1(:, 2:2), 1, i, nao, 1, 1.0_dp, 1.0_dp)
            endif
          enddo
        enddo
        call dbcsr_deallocate_matrix(pmat%matrix)
        
        ! this is debug to make sure active-active part is correct >>>>>>
        call cp_fm_struct_create(mo_mo_fm_struct, nrow_global=nmo, ncol_global=nmo, context=blacs_env)
        call cp_fm_create(ortho_fm, mo_mo_fm_struct)
        call cp_fm_struct_release(mo_mo_fm_struct)
        call cp_fm_gemm('T', 'N', nmo, nmo, nao, 1.0_dp, mo_coeff, ztrans(ispin)%matrix, 0.0_dp, ortho_fm)
        call cp_fm_write_formatted(ortho_fm, iw, 'ACTIVE_SPACE')
        call cp_fm_release(ortho_fm)
        ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      enddo
      deallocate(col1)
      call deallocate_task_list(task_list_sub)
      call set_qs_env(qs_env, force=force)

      call cp_fm_release(matrix_ks_fm)
      call cp_fm_release(work_fm)
      call cp_fm_release(col_fm)
      call cp_fm_release(col2_fm)

      CALL pw_pool_give_back_pw(auxbas_pw_pool, pot_r%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, pot_g%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_r%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_g%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_tot_g%pw)

      ! restore the initial value of the cutoff
      CALL pw_env_release(pw_env_sub)
      dft_control%qs_control%e_cutoff = e_cutoff_old
      dft_control%qs_control%cutoff = cutoff_old
      dft_control%qs_control%relative_cutoff = relative_cutoff_old
      deallocate(e_cutoff_old)

      ! next we calculate the total Fock operator
      CALL dbcsr_allocate_matrix_set(active_space_env%pmat_total, nspins)
      do ispin = 1, nspins
         call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=mo_coeff2, nmo=nmo)
         allocate(denmat_ref)
         call dbcsr_copy(denmat_ref, active_space_env%pmat_reference(ispin)%matrix)
         call cp_gemm('N', 'N', nao, nmo, nmo, 1.0_dp, mo_coeff2, rdm1_fm(ispin)%ptr, 0.0_dp, tmp_fm)
         call cp_dbcsr_plus_fm_fm_t(denmat_ref, mo_coeff2, tmp_fm, nmo)
         active_space_env%pmat_total(ispin)%matrix => denmat_ref
      enddo
      call cp_fm_release(tmp_fm)
      call calculate_fock(qs_env, active_space_env%pmat_total)
      call get_qs_env(qs_env, matrix_ks=matrix_ks)
      allocate(kstot_fm(nspins))
      do ispin = 1, nspins
        call cp_fm_create(kstot_fm(ispin)%matrix, ao_ao_fm_struct)
        call copy_dbcsr_to_fm(matrix_ks(ispin)%matrix, kstot_fm(ispin)%matrix)
      enddo

      ! we need to revert to the original qs_env.
      ! This is needed since I need the hf fock matrix in calculate_energy_deriv
      call restore_qs_env(qs_env, active_space_env)

      call create_projectors(qs_env, active_space_env, naocc)
      call calculate_energy_deriv(qs_env, active_space_env, kstot_fm, ztrans)
      call calculate_z_matrix(qs_env, active_space_env, eigenvalues, naocc, active_orb_list, nactive_orb)
      deallocate(eigenvalues)
      allocate(zmos(nspins))
      call get_response_zmos(qs_env, active_space_env, nactive_orb, active_orb_list, kstot_fm, zmos)

      ! before entering response equation, we need to revert to the original qs_env...
      call restore_qs_env(qs_env, active_space_env)
      call get_qs_env(qs_env, energy=energy)
      if (iw > 0) then
        write(iw, *) "recalculated ", energy%total
      endif

      nullify(p_env)
      NULLIFY (lr_section)
      lr_section => section_vals_get_subs_vals(as_input, "CPHF")
      call response_equation(qs_env, p_env, zmos, iw, lr_section=lr_section)

      if (nspins > 1) then
        ! Why did i forget somewhere else a factor? Probably in get_response_zmos
        factor = 2.0_dp
      else
        factor = 2.0_dp
      endif
      do ispin = 1, nspins
        call dbcsr_add(active_space_env%pmat_z(ispin)%matrix, p_env%p1(ispin)%matrix, 1.0_dp, factor)
        call dbcsr_add(active_space_env%pmat_x(ispin)%matrix, p_env%w1(ispin)%matrix, 1.0_dp, factor)
        call cp_fm_release(zmos(ispin)%matrix)
      enddo
      call p_env_release(p_env)
      call calculate_x_matrix(qs_env, active_space_env, kstot_fm, naocc, active_orb_list, nactive_orb)

      do ispin = 1, nspins
        call cp_fm_release(ztrans(ispin)%matrix)
        call cp_fm_release(kstot_fm(ispin)%matrix)
      enddo
      deallocate(ztrans, kstot_fm, zmos)

      call restore_qs_env(qs_env, active_space_env)
      call get_qs_env(qs_env, rho=rho)
      call qs_rho_get(rho, rho_ao=rho_ao)

      ! call release_active_space_type(qs_env%active_space)
      call set_qs_env(qs_env, active_space=active_space_env)

      ! debugging
      do ispin = 1, nspins
        ! inactive contribution
        call dbcsr_copy(rho_ao(ispin)%matrix, active_space_env%pmat_total(ispin)%matrix)
        ! zvector contribution
        call dbcsr_add(rho_ao(ispin)%matrix, active_space_env%pmat_z(ispin)%matrix, 1.0_dp, 1.0_dp)
      enddo
      call qs_rho_update_rho(rho, qs_env) !just in case for consistency...
      call qs_ks_did_change(ks_env, rho_changed=.true.)
      call qs_moment_locop(qs_env, .false., 1, use_mom_ref_coac, ref_point, iw)
      call restore_qs_env(qs_env, active_space_env)
      energy%total = active_space_env%energy_total


      ! release active RDMs
      nmo = active_space_env%eri%norb
      do ispin = 1, nspins
        call cp_fm_release(rdm1_fm(ispin)%ptr)
        do i = 1, nmo
          do j = 1, nmo
            if (associated(rdm2_fm(ispin, i, j)%ptr)) then
              call cp_fm_release(rdm2_fm(ispin, i, j)%ptr)
            endif
          enddo
        enddo
      enddo
      deallocate(rdm1_fm)
      deallocate(rdm2_fm)

      CALL cp_fm_struct_release(ao_fm_struct)
      CALL cp_fm_struct_release(ao_ao_fm_struct)
      CALL cp_fm_struct_release(ao_mo_fm_struct)
      CALL cp_fm_struct_release(mo_ao_fm_struct)
      qs_env%requires_mo_derivs = requires_mo_derivs_old

      IF (iw > 0) THEN
         WRITE (iw, '(/,T2,A)') &
            '!-------------------- End of Active Space Interface --------------------------!'
      END IF

      CALL timestop(handle)

   END SUBROUTINE active_space_main

! **************************************************************************************************
!> \brief Calculate ediv(ispin)_alpha,j = S * C_alpha,i * dE/dU_i,j for j in core or active space

!> \param qs_env ...
!> \param active_space_env ...
!> \param kstot_fm ...
!> \param ztrans ...
!> \param Y ...
! **************************************************************************************************
   SUBROUTINE calculate_energy_deriv(qs_env, active_space_env, kstot_fm, ztrans)
      TYPE(qs_environment_type), POINTER, intent(in)        :: qs_env
      TYPE(active_space_type), POINTER, intent(inout)       :: active_space_env
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER, intent(in) :: kstot_fm, ztrans

      CHARACTER(len=*), PARAMETER                        :: routineN = 'calculate_energy_deriv'

      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), POINTER                          :: core_coeff

      REAL(KIND=dp) :: factor
      INTEGER :: nao, ncore, nact, nspins, ispin, ntot

      call get_qs_env(qs_env, blacs_env=blacs_env)
      nspins = active_space_env%nspins

      if (nspins > 1) then
        factor = 1.0_dp
      else
        factor = 2.0_dp
      endif

      allocate (active_space_env%ediv(nspins))
      do ispin = 1, nspins
        call get_mo_set(active_space_env%mos_inactive(ispin)%mo_set, mo_coeff=core_coeff, nao=nao, nmo=ncore)
        nact = ztrans(ispin)%matrix%matrix_struct%ncol_global
        ntot = nact+ncore

        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=ntot, context=blacs_env)
        call cp_fm_create(active_space_env%ediv(ispin)%matrix, fm_struct)
        call cp_fm_struct_release(fm_struct)

        call cp_gemm('N', 'N', nao, ncore, nao, factor, kstot_fm(ispin)%matrix, core_coeff, 0.0_dp,&
                     active_space_env%ediv(ispin)%matrix)
        call cp_fm_to_fm(ztrans(ispin)%matrix, active_space_env%ediv(ispin)%matrix, nact, 1, ncore+1)
      enddo

   END SUBROUTINE calculate_energy_deriv


! **************************************************************************************************
!> \brief calculates the lagrangian wrt 'core - active occ' and 'active virt - external' rotations
!> \param input ...
!> \param logger ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE calculate_z_matrix(qs_env, active_space_env, eigenvalues, naocc, active_orb_list, nactive_orb)
      TYPE(qs_environment_type), POINTER, intent(in)     :: qs_env
      TYPE(active_space_type), POINTER, intent(inout)    :: active_space_env
      REAL(KIND=dp), DIMENSION(:, :), intent(in)         :: eigenvalues
      INTEGER, DIMENSION(2), intent(in)                  :: naocc, nactive_orb
      INTEGER, DIMENSION(:, :), intent(in)               :: active_orb_list

      CHARACTER(len=*), PARAMETER                        :: routineN = 'calculate_z_matrix'

      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), POINTER                          :: core_coeff, act_coeff, work_fm, work2_fm,&
                                                            z1_fm, z2_fm, ks_hf_fm, col_fm, virt_coeff, &
                                                            cholesky_fm, ediv
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s, matrix_t, ks_hf

      REAL(KIND=dp), DIMENSION(:), ALLOCATABLE :: core_eigenvalues, oact_eigenvalues, vact_eigenvalues
      REAL(KIND=dp) :: el
      LOGICAL :: in_active
      INTEGER :: nao, ncore, nact, nvirt, nspins, ispin, i, j, iaocc, ivirt, icore, homo

      call get_qs_env(qs_env, blacs_env=blacs_env, matrix_s=matrix_s, matrix_ks=ks_hf, kinetic=matrix_t)
      nspins = active_space_env%nspins
      call dbcsr_allocate_matrix_set(active_space_env%pmat_z, nspins)
      call dbcsr_allocate_matrix_set(active_space_env%pmat_x, nspins)

      call get_mo_set(active_space_env%mos_active(1)%mo_set, nao=nao)
      call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=nao, context=blacs_env)
      call cp_fm_create(cholesky_fm, fm_struct)
      call cp_fm_struct_release(fm_struct)
!     call copy_dbcsr_to_fm(matrix_t(1)%matrix, cholesky_fm)
!     call cp_fm_cholesky_decompose(cholesky_fm)
!     call cp_fm_cholesky_invert(cholesky_fm)

      do ispin = 1, nspins
        allocate(active_space_env%pmat_z(ispin)%matrix)
        allocate(active_space_env%pmat_x(ispin)%matrix)
        call dbcsr_copy(active_space_env%pmat_z(ispin)%matrix, active_space_env%pmat_total(ispin)%matrix,&
                        name="LAGRANGIAN MULTIPLIER Z")
        call dbcsr_copy(active_space_env%pmat_x(ispin)%matrix, active_space_env%pmat_total(ispin)%matrix,&
                        name="LAGRANGIAN MULTIPLIER X")
        call dbcsr_set(active_space_env%pmat_z(ispin)%matrix, 0.0_dp)
        call dbcsr_set(active_space_env%pmat_x(ispin)%matrix, 0.0_dp)
        ediv => active_space_env%ediv(ispin)%matrix

        call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=act_coeff, nao=nao, nmo=nact)
        call get_mo_set(active_space_env%mos_inactive(ispin)%mo_set, mo_coeff=core_coeff, nmo=ncore)

        call cp_fm_struct_create(fm_struct, nrow_global=naocc(ispin), ncol_global=ncore, context=blacs_env)
        call cp_fm_create(z1_fm, fm_struct)
        call cp_fm_create(z2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        nvirt = nact - naocc(ispin)
        homo = ncore + naocc(ispin)

        ! Fill in eigenvalues
        allocate(core_eigenvalues(ncore), oact_eigenvalues(naocc(ispin)), vact_eigenvalues(nvirt))
        iaocc = 0
        ivirt = 0
        icore = 0
        !! Should probably create a mapping!
        do i = 1, nactive_orb(ispin)
          j = active_orb_list(i, ispin)
          if (j <= homo) then
            iaocc = iaocc + 1
            oact_eigenvalues(iaocc) = eigenvalues(j, ispin)
          else
            ivirt = ivirt + 1
            vact_eigenvalues(ivirt) = eigenvalues(j, ispin)
          endif
        enddo
        do i = 1, homo
          in_active = .false.
          do j = 1, nactive_orb(ispin)
            if (i == active_orb_list(j, ispin)) then
              in_active = .true.
              exit
            endif
          enddo
          if (.not. in_active) then
            icore = icore + 1
            core_eigenvalues(icore) = eigenvalues(i, ispin)
          endif
        enddo
        CPASSERT(icore == ncore)
        CPASSERT(ivirt == nvirt)
        CPASSERT(iaocc == naocc(ispin))

        call cp_fm_create(work_fm, core_coeff%matrix_struct)

        ! occupied active with occupied projection, transposed, hence a minus sign
        call cp_gemm('T', 'N', naocc(ispin), ncore, nao, 1.0_dp, act_coeff, ediv, 0.0_dp, z1_fm)
        call cp_gemm('T', 'N', naocc(ispin), ncore, nao, 1.0_dp, ediv, core_coeff, 0.0_dp, z2_fm, ncore+1, 1, 1, 1, 1, 1)

        ! divide by (epsilon_i - epsilon_j)
        do i = 1, naocc(ispin)
          CPASSERT(i <= nactive_orb(ispin))
          do j = 1, ncore
            call cp_fm_get_element(z1_fm, i, j, el)
            el = -el / (oact_eigenvalues(i) - core_eigenvalues(j))
            call cp_fm_set_element(z1_fm, i, j, el)

            call cp_fm_get_element(z2_fm, i, j, el)
            el = el / (oact_eigenvalues(i) - core_eigenvalues(j))
            call cp_fm_set_element(z2_fm, i, j, el)
          enddo
        enddo

        ! Transform to ao
        call cp_gemm('N', 'N', nao, ncore, naocc(ispin), 1.0_dp, act_coeff, z1_fm, 0.0_dp, work_fm)
        call cp_gemm('N', 'N', nao, ncore, naocc(ispin), 1.0_dp, act_coeff, z2_fm, 1.0_dp, work_fm)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_z(ispin)%matrix, work_fm, core_coeff, ncore)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_z(ispin)%matrix, core_coeff, work_fm, ncore)

        ! z density * Fock_HF
        call cp_fm_column_scale(z1_fm, core_eigenvalues)
        call cp_fm_row_scale(z2_fm, oact_eigenvalues)

        call cp_gemm('N', 'N', nao, ncore, naocc(ispin), 1.0_dp, act_coeff, z1_fm, 0.0_dp, work_fm)
        call cp_gemm('N', 'N', nao, ncore, naocc(ispin), 1.0_dp, act_coeff, z2_fm, 1.0_dp, work_fm)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_x(ispin)%matrix, work_fm, core_coeff, ncore)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_x(ispin)%matrix, core_coeff, work_fm, ncore)
        call cp_fm_release(z1_fm)
        call cp_fm_release(z2_fm)
        call cp_fm_release(work_fm)

        !! For virt-external rotations
        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=1, context=blacs_env)
        call cp_fm_create(col_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)

        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=nvirt, context=blacs_env)
        call cp_fm_create(z1_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)

        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=nao, context=blacs_env)
        call cp_fm_create(work_fm, fm_struct)
        call cp_fm_create(ks_hf_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call copy_dbcsr_to_fm(ks_hf(ispin)%matrix, ks_hf_fm)

        ! lapack version - project to virtual here; also add tiny bit of diagonal to null space
        call copy_dbcsr_to_fm(active_space_env%project_external(ispin)%matrix, cholesky_fm)
        call cp_gemm("N", "N", nao, nao, nao, 1.0_dp, cholesky_fm, ks_hf_fm, 0.0_dp, work_fm) 
        call cp_gemm("N", "T", nao, nao, nao, 1.0_dp, work_fm, cholesky_fm, 0.0_dp, ks_hf_fm)

        ! Don't know if this is even needed
        ! Setting the full active and core space probably non-degenerate to to the external space
        call cp_fm_create(work2_fm, act_coeff%matrix_struct)
        call cp_dbcsr_sm_fm_multiply(matrix_s(1)%matrix, act_coeff, work2_fm, nact)
        call cp_gemm("N", "T", nao, nao, nact, -10.0_dp, work2_fm, work2_fm, 1.0_dp, ks_hf_fm)
        call cp_fm_release(work2_fm)

        call cp_fm_create(work2_fm, core_coeff%matrix_struct)
        call cp_dbcsr_sm_fm_multiply(matrix_s(1)%matrix, core_coeff, work2_fm, ncore)
        call cp_gemm("N", "T", nao, nao, ncore, -10.0_dp, work2_fm, work2_fm, 1.0_dp, ks_hf_fm)
        call cp_fm_release(work2_fm)

        ! Loop over different act virtual orbitals
        do i = 1, nvirt
          call cp_fm_set_all(work_fm, 0.0_dp)
          call copy_dbcsr_to_fm(matrix_s(1)%matrix, work_fm)
          call cp_fm_scale_and_add(vact_eigenvalues(i), work_fm, -1.0_dp, ks_hf_fm)
          call cp_fm_to_fm(ediv, col_fm, 1, i + ncore + naocc(ispin), 1)
          !call my_fm_solve(work_fm, col_fm, cholesky_fm, active_space_env%project_external(ispin)%matrix)
          !lapack versions
          call cp_fm_solve(work_fm, col_fm)
          ! Project onto external one last time to be sure
          call cp_gemm("T", "N", nao, 1, nao, 1.0_dp, cholesky_fm, col_fm, 0.0_dp, z1_fm, c_first_col=i)
        enddo
        call cp_fm_release(work_fm)

        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=nvirt, context=blacs_env)
        call cp_fm_create(virt_coeff, fm_struct)
        call cp_fm_struct_release(fm_struct)

        call cp_fm_to_fm(act_coeff, virt_coeff, nvirt, naocc(ispin)+1, 1)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_z(ispin)%matrix, z1_fm, virt_coeff, nvirt)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_z(ispin)%matrix, virt_coeff, z1_fm, nvirt)

        call cp_fm_column_scale(z1_fm, vact_eigenvalues)

        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_x(ispin)%matrix, z1_fm, virt_coeff, nvirt)
        call cp_dbcsr_plus_fm_fm_t(active_space_env%pmat_x(ispin)%matrix, virt_coeff, z1_fm, nvirt)

        call cp_fm_release(col_fm)
        call cp_fm_release(z1_fm)
        call cp_fm_release(ks_hf_fm)
        call cp_fm_release(virt_coeff)
        deallocate(core_eigenvalues, oact_eigenvalues, vact_eigenvalues)
      enddo
      call cp_fm_release(cholesky_fm)
   END SUBROUTINE calculate_z_matrix


! **************************************************************************************************
!> \brief calculates the lagrangian wrt orbital orthogonality
!> \param input ...
!> \param logger ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE calculate_x_matrix(qs_env, active_space_env, kstot_fm, naocc, active_orb_list, nactive_orb)
      TYPE(qs_environment_type), POINTER, intent(in)     :: qs_env
      TYPE(active_space_type), POINTER, intent(inout)    :: active_space_env
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER, intent(in) :: kstot_fm
      INTEGER, DIMENSION(2), intent(in)                  :: naocc, nactive_orb
      INTEGER, DIMENSION(:, :), intent(in)               :: active_orb_list

      CHARACTER(len=*), PARAMETER                        :: routineN = 'calculate_x_matrix'

      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), POINTER                          :: mo_hf, mo_core, mo_act, kstot2_fm, work_fm, work2_fm
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao, matrix_ks
      TYPE(dbcsr_type), POINTER                          :: X, rho_z
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos
      INTEGER :: nao, ncore, nact, nvirt, nspins, ispin, nmo, i, j, iaocc
      REAL(KIND=dp) :: factor
      REAL(KIND=dp), DIMENSION(:), POINTER :: eigenvalues
      REAL(KIND=dp), DIMENSION(:), ALLOCATABLE :: oact_eigenvalues

      call get_qs_env(qs_env, blacs_env=blacs_env, mos=mos)
      nspins = active_space_env%nspins

      if (nspins > 1) then
        factor = 1.0_dp
      else
        factor = 2.0_dp
      endif

      ! ############################################################
      ! Calculate total fock for all rho's
      !! TODO change this, I just do this as a quick check
      nullify(rho_ao)
      call dbcsr_allocate_matrix_set(rho_ao, nspins)
      do ispin = 1, nspins
        allocate(rho_ao(ispin)%matrix)
        call dbcsr_copy(rho_ao(ispin)%matrix, active_space_env%pmat_total(ispin)%matrix)
        call dbcsr_add(rho_ao(ispin)%matrix, active_space_env%pmat_z(ispin)%matrix, 1.0_dp, 1.0_dp)
      enddo

      call calculate_fock(qs_env, rho_ao)
      call get_qs_env(qs_env, matrix_ks=matrix_ks)
      call dbcsr_deallocate_matrix_set(rho_ao)
      ! ############################################################

      call cp_fm_create(kstot2_fm, kstot_fm(1)%matrix%matrix_struct)
      do ispin = 1, nspins
        call get_mo_set(mo_set=mos(ispin)%mo_set, mo_coeff=mo_hf, nmo=nmo, nao=nao, eigenvalues=eigenvalues)
        call get_mo_set(mo_set=active_space_env%mos_inactive(ispin)%mo_set, mo_coeff=mo_core, nmo=ncore)
        call get_mo_set(mo_set=active_space_env%mos_active(ispin)%mo_set, mo_coeff=mo_act, nmo=nact)
        nvirt = nact-naocc(ispin)

        !! rho_z * fock contribution is already stored in pmat_x
        X => active_space_env%pmat_x(ispin)%matrix
        rho_z => active_space_env%pmat_z(ispin)%matrix

        !! 2 * P_occ * g(rho_z) * P_occ
        call cp_fm_struct_create(fm_struct, nrow_global=nmo, ncol_global=nmo, context=blacs_env)
        call cp_fm_create(work2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call cp_fm_create(work_fm, mo_hf%matrix_struct)

        call cp_fm_set_all(kstot2_fm, 0.0_dp)
        call copy_dbcsr_to_fm(matrix_ks(ispin)%matrix, kstot2_fm)
        call cp_fm_scale_and_add(1.0_dp, kstot2_fm, -1.0_dp, kstot_fm(ispin)%matrix)
        call cp_gemm('N', 'N', nao, nmo, nao, 1.0_dp, kstot2_fm, mo_hf, 0.0_dp, work_fm)
        call cp_gemm('T', 'N', nmo, nmo, nao, 1.0_dp, work_fm, mo_hf, 0.0_dp, work2_fm)
        call cp_gemm('N', 'N', nao, nmo, nmo, 1.0_dp, mo_hf, work2_fm, 0.0_dp, work_fm)
        call cp_dbcsr_plus_fm_fm_t(X, work_fm, mo_hf, nmo, alpha=factor)
        call cp_fm_release(work_fm)
        call cp_fm_release(work2_fm)

        !! symmetric core-core
        call cp_fm_struct_create(fm_struct, nrow_global=ncore, ncol_global=ncore, context=blacs_env)
        call cp_fm_create(work2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call cp_fm_create(work_fm, mo_core%matrix_struct)
        call cp_gemm('T', 'N', ncore, ncore, nao, 1.0_dp, mo_core, active_space_env%ediv(ispin)%matrix, 0.0_dp, work2_fm)
        call cp_gemm('N', 'N', nao, ncore, ncore, 1.0_dp, mo_core, work2_fm, 0.0_dp, work_fm)
        call cp_dbcsr_plus_fm_fm_t(X, work_fm, mo_core, ncore)
        call cp_fm_release(work2_fm)
        call cp_fm_release(work_fm)

        !! symmetric active-active (reference)
        iaocc = 0
        allocate(oact_eigenvalues(naocc(ispin)))
        do i = 1, nactive_orb(ispin)
          j = active_orb_list(i, ispin)
          if (j <= nmo) then
            iaocc = iaocc + 1
            oact_eigenvalues(iaocc) = eigenvalues(j)
          endif
        enddo
        CPASSERT(iaocc == naocc(ispin))

        call cp_fm_struct_create(fm_struct, nrow_global=nao, ncol_global=nmo-ncore, context=blacs_env)
        call cp_fm_create(work_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call cp_fm_to_fm(mo_act, work_fm, nmo-ncore, 1, 1)
        call cp_fm_column_scale(work_fm, oact_eigenvalues)
        call cp_dbcsr_plus_fm_fm_t(X, work_fm, mo_act, nmo-ncore, alpha=factor)
        call cp_fm_release(work_fm)
        deallocate(oact_eigenvalues)

        !! symmetric active-active
        call cp_fm_struct_create(fm_struct, nrow_global=nact, ncol_global=nact, context=blacs_env)
        call cp_fm_create(work2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call cp_fm_create(work_fm, mo_act%matrix_struct)

        call cp_gemm('T', 'N', nact, nact, nao, 1.0_dp, mo_act, active_space_env%ediv(ispin)%matrix, 0.0_dp, work2_fm,&
                     1, 1, ncore+1, 1, 1, 1)
        call cp_gemm('N', 'N', nao, nact, nact, 1.0_dp, mo_act, work2_fm, 0.0_dp, work_fm)
        call cp_dbcsr_plus_fm_fm_t(X, work_fm, mo_act, nact)
        call cp_fm_release(work2_fm)
        call cp_fm_release(work_fm)

        !! core-virtual
        call cp_fm_struct_create(fm_struct, nrow_global=ncore, ncol_global=nvirt, context=blacs_env)
        call cp_fm_create(work2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)
        call cp_fm_create(work_fm, mo_core%matrix_struct)

        call cp_gemm('T', 'N', ncore, nvirt, nao, 1.0_dp, mo_core, active_space_env%ediv(ispin)%matrix, 0.0_dp, work2_fm, &
                     b_first_col=ncore+naocc(ispin)+1)
        call cp_gemm('N', 'T', nao, ncore, nvirt, 1.0_dp, mo_act, work2_fm, 0.0_dp, work_fm, a_first_col=naocc(ispin)+1)
        call cp_dbcsr_plus_fm_fm_t(X, work_fm, mo_core, ncore)
        call cp_dbcsr_plus_fm_fm_t(X, mo_core, work_fm, ncore)
        call cp_fm_release(work2_fm)
        call cp_fm_release(work_fm)

        !! For UHF still the case?
        call dbcsr_scale(X, 2.0_dp)
      enddo
      call cp_fm_release(kstot2_fm)
   END SUBROUTINE calculate_x_matrix

! **************************************************************************************************
!> \brief solves linear equation using iterative algorithm Ax = b
!> \param work_fm
!> \param col_fm ...
! toru toru
! **************************************************************************************************
   SUBROUTINE my_fm_solve(a, b, chol, proj)

      TYPE(cp_fm_type), POINTER, INTENT(in)              :: a
      TYPE(cp_fm_type), POINTER, INTENT(inout)           :: b
      TYPE(cp_fm_type), POINTER, INTENT(in)              :: chol
      TYPE(dbcsr_type), POINTER, INTENT(in)              :: proj

      TYPE(cp_fm_p_type)                                 :: c, t, r, res
      TYPE(cp_fm_type), POINTER                          :: tmp
      TYPE(dbcsr_type)                                   :: projT
      integer       :: maxiter, maxsubspace, i, m, iw
      real(kind=dp) :: norm

      CPASSERT(associated(proj))

      ! proj  : S CC^+
      ! projT : CC^+ S
      call dbcsr_copy(projT, proj)
      call dbcsr_transposed(projT, proj)

      maxiter = 100
      maxsubspace = 20
      m = a%matrix_struct%ncol_global
      CPASSERT(m == a%matrix_struct%nrow_global)
      CPASSERT(m == b%matrix_struct%nrow_global)
      iw = cp_logger_get_default_io_unit(cp_get_default_logger())

      call cp_fm_create(tmp, b%matrix_struct)

      nullify(c%matrix)
      call cp_fm_create(c%matrix, b%matrix_struct)
      call cp_dbcsr_sm_fm_multiply(proj, b, c%matrix, 1, -1.0_dp)
      call initialize_subspace_linear(maxsubspace, c)

      ! initial vector is the source vector
      nullify(t%matrix)
      call cp_fm_create(t%matrix, b%matrix_struct)
      call cp_dbcsr_sm_fm_multiply(proj, b, t%matrix, 1)
      call cp_fm_symm("L", "U", m, 1, 1.0_dp, chol, t%matrix, 0.0_dp, tmp)
      call cp_dbcsr_sm_fm_multiply(projT, tmp, t%matrix, 1)

      if (iw > 0) then 
        write(iw, *) "status"
      endif
      do i = 1, maxiter
        ! normalize
        call cp_fm_trace(t%matrix, t%matrix, norm)
        call cp_fm_scale(1.0_dp/sqrt(norm), t%matrix)

        nullify(r%matrix)
        call cp_fm_create(r%matrix, b%matrix_struct)
        call cp_fm_symm("L", "U", m, 1, 1.0_dp, a, t%matrix, 0.0_dp, tmp)
        call cp_dbcsr_sm_fm_multiply(proj, tmp, r%matrix, 1)

        nullify(res%matrix)
        call cp_fm_create(res%matrix, b%matrix_struct)
        call update_subspace_linear(t, r, res)

        call cp_fm_trace(res%matrix, res%matrix, norm)

        nullify(t%matrix)
        call cp_fm_create(t%matrix, b%matrix_struct)

        if (iw > 0) then 
          write(iw, '(I4, F25.10)') i, sqrt(norm)
        endif
        if (sqrt(norm) < 1.0e-6) then
          call cp_fm_release(res%matrix)
          exit
        else
          call cp_dbcsr_sm_fm_multiply(proj, res%matrix, t%matrix, 1)
          call cp_fm_symm("L", "U", m, 1, 1.0_dp, chol, t%matrix, 0.0_dp, res%matrix)
          call cp_dbcsr_sm_fm_multiply(projT, res%matrix, t%matrix, 1)
          call cp_fm_release(res%matrix)
        endif
      enddo

      call solution_subspace_linear(t)
      call cp_dbcsr_sm_fm_multiply(projT, t%matrix, b, 1)
      call cp_fm_release(t%matrix) ! everything else is deallocated in the solver
      call terminate_subspace_linear()

      call cp_fm_release(tmp)
      call dbcsr_release(projT)

   END SUBROUTINE my_fm_solve

! **************************************************************************************************
!> \brief Creates projectors relevant to the active space
!> \param input ...
!> \param logger ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE create_projectors(qs_env, active_space_env, naocc)

      TYPE(qs_environment_type), POINTER, intent(in)     :: qs_env
      TYPE(active_space_type), POINTER, intent(inout)    :: active_space_env
      INTEGER, DIMENSION(2), intent(in)                  :: naocc

      CHARACTER(len=*), PARAMETER                        :: routineN = 'create_projectors'

      TYPE(cp_fm_struct_type), POINTER                   :: ao_ao_fm_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), POINTER                          :: core_coeff, act_coeff, unit_matrix_fm
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s
      TYPE(dbcsr_type), POINTER                          :: proj_virt, proj_ext
      TYPE(dbcsr_type)                                   :: db_temp

      INTEGER :: nao, ncore, nact, nspins, ispin

      call get_qs_env(qs_env, blacs_env=blacs_env, matrix_s=matrix_s)
      nspins = active_space_env%nspins

      call dbcsr_allocate_matrix_set(active_space_env%project_virtual, nspins)
      call dbcsr_allocate_matrix_set(active_space_env%project_external, nspins)
      do ispin = 1, nspins
        call get_mo_set(active_space_env%mos_inactive(ispin)%mo_set, mo_coeff=core_coeff, nmo=ncore)
        call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=act_coeff, nao=nao, nmo=nact)
        call cp_fm_struct_create(ao_ao_fm_struct, nrow_global=nao, ncol_global=nao, context=blacs_env)

        allocate(active_space_env%project_virtual(ispin)%matrix)
        allocate(active_space_env%project_external(ispin)%matrix)
        proj_virt => active_space_env%project_virtual(ispin)%matrix
        proj_ext => active_space_env%project_external(ispin)%matrix
        ! desymmetrize
        call dbcsr_desymmetrize(matrix_s(1)%matrix, proj_virt)
        call dbcsr_desymmetrize(matrix_s(1)%matrix, proj_ext)

        call cp_fm_create(unit_matrix_fm, ao_ao_fm_struct)
        call cp_fm_set_all(unit_matrix_fm, 0.0_dp, 1.0_dp)
        call copy_fm_to_dbcsr(unit_matrix_fm, proj_virt)
        call copy_fm_to_dbcsr(unit_matrix_fm, proj_ext)
        call cp_fm_release(unit_matrix_fm)

        !! db_temp is a symmetric matrix since matrix_s is also symmetric
        call dbcsr_copy(db_temp, matrix_s(1)%matrix)
        call dbcsr_set(db_temp, 0.0_dp)
        call cp_dbcsr_plus_fm_fm_t(db_temp, core_coeff, core_coeff, ncore)
        call cp_dbcsr_plus_fm_fm_t(db_temp, act_coeff, act_coeff, naocc(ispin))
        call dbcsr_multiply('N', 'N', -1.0_dp, matrix_s(1)%matrix, db_temp, 1.0_dp, proj_virt)

        call dbcsr_set(db_temp, 0.0_dp)
        call cp_dbcsr_plus_fm_fm_t(db_temp, core_coeff, core_coeff, ncore)
        call cp_dbcsr_plus_fm_fm_t(db_temp, act_coeff, act_coeff, nact)
        call dbcsr_multiply('N', 'N', -1.0_dp, matrix_s(1)%matrix, db_temp, 1.0_dp, proj_ext)
        call dbcsr_release(db_temp)
        call cp_fm_struct_release(ao_ao_fm_struct)
      enddo
   END SUBROUTINE create_projectors


! **************************************************************************************************
!> \brief Update ediv_asym for the contribution for the z_matrix
!> \param input ...
!> \param logger ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE get_response_zmos(qs_env, active_space_env, nactive_orb, active_orb_list, kstot_fm, zmos)

      TYPE(qs_environment_type), POINTER, intent(in)     :: qs_env
      TYPE(active_space_type), POINTER, intent(in)       :: active_space_env
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER, intent(in) :: kstot_fm
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER, intent(inout) :: zmos
      INTEGER, DIMENSION(2), intent(in)                  :: nactive_orb
      INTEGER, DIMENSION(:, :), intent(in)               :: active_orb_list

      CHARACTER(len=*), PARAMETER                        :: routineN = 'get_response_zmos'

      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), POINTER                          :: core_coeff, act_coeff, kstot2_fm, hf_coeff,&
                                                            tmp_fm, tmp2_fm, work_fm, full_coeff, ediv
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao, matrix_ks, matrix_s
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos

      LOGICAL :: in_active
      REAL(KIND=dp) :: factor
      INTEGER :: nao, ncore, nact, ntot, nspins, ispin, k, i, j, noact, nvact

      call get_qs_env(qs_env, mos=mos, matrix_s=matrix_s, blacs_env=blacs_env)
      nspins = active_space_env%nspins

      CPASSERT(ASSOCIATED(zmos))

      ! We calculate the total Fock operator (again)
      ! in essence I only need the potential due to the pmat_z density.
      ! TODO: change this
      nullify(rho_ao)
      call dbcsr_allocate_matrix_set(rho_ao, nspins)
      do ispin = 1, nspins
        allocate(rho_ao(ispin)%matrix)
        call dbcsr_copy(rho_ao(ispin)%matrix, active_space_env%pmat_total(ispin)%matrix)
        call dbcsr_add(rho_ao(ispin)%matrix, active_space_env%pmat_z(ispin)%matrix, 1.0_dp, 1.0_dp)
      enddo

      call calculate_fock(qs_env, rho_ao)
      call get_qs_env(qs_env, matrix_ks=matrix_ks)
      call dbcsr_deallocate_matrix_set(rho_ao)

      if (nspins > 1) then
        factor = 1.0_dp
      else
        factor = 2.0_dp
      endif

      call cp_fm_create(kstot2_fm, kstot_fm(1)%matrix%matrix_struct)
      do ispin = 1, nspins
        ediv => active_space_env%ediv(ispin)%matrix
        call get_mo_set(mo_set=mos(ispin)%mo_set, mo_coeff=hf_coeff, nmo=ntot)
        call get_mo_set(active_space_env%mos_inactive(ispin)%mo_set, mo_coeff=core_coeff, nmo=ncore)
        call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=act_coeff, nao=nao, nmo=nact)
        noact = ntot - ncore
        nvact = nact - noact

        call cp_fm_create(work_fm, hf_coeff%matrix_struct)
        call cp_fm_create(tmp_fm, hf_coeff%matrix_struct)
        call cp_fm_create(zmos(ispin)%matrix, hf_coeff%matrix_struct)
        call cp_fm_create(full_coeff, hf_coeff%matrix_struct)
        call cp_fm_to_fm(core_coeff, full_coeff, ncore, 1, 1)
        call cp_fm_to_fm(act_coeff, full_coeff, noact, 1, ncore+1)

        call cp_fm_struct_create(fm_struct, nrow_global=nvact, ncol_global=ntot, context=blacs_env)
        call cp_fm_create(tmp2_fm, fm_struct)
        call cp_fm_struct_release(fm_struct)

        call cp_fm_to_fm(ediv, work_fm, ntot)

        !! Other parts of ediv.T will be projected out
        call cp_gemm('T', 'N', nvact, ntot, nao, 1.0_dp, ediv, full_coeff, 0.0_dp, tmp2_fm, a_first_col=ntot+1)
        call cp_gemm('N', 'N', nao, ntot, nvact, 1.0_dp, act_coeff, tmp2_fm, 0.0_dp, tmp_fm, a_first_col=noact+1)
        call cp_dbcsr_sm_fm_multiply(matrix_s(1)%matrix, tmp_fm, work_fm, ntot, alpha=-1.0_dp, beta=1.0_dp)

        call cp_fm_set_all(kstot2_fm, 0.0_dp)
        call copy_dbcsr_to_fm(matrix_ks(ispin)%matrix, kstot2_fm)
        call cp_fm_scale_and_add(1.0_dp, kstot2_fm, -1.0_dp, kstot_fm(ispin)%matrix)
        call cp_gemm('N', 'N', nao, ntot, nao, factor, kstot2_fm, full_coeff, 1.0_dp, work_fm)

        call cp_dbcsr_sm_fm_multiply(active_space_env%project_virtual(ispin)%matrix, work_fm, tmp_fm, ntot)
        call cp_fm_release(work_fm)
        call cp_fm_release(tmp2_fm)
        call cp_fm_release(full_coeff)

        ! Correctly order zmos
        k = 0
        do i = 1, ntot
          in_active = .False.
          do j = 1, nactive_orb(ispin)
            if (i == active_orb_list(j, ispin)) then
              call cp_fm_to_fm(tmp_fm, zmos(ispin)%matrix, 1, j+ncore, i)
              in_active = .True.
              exit
            endif
          enddo
          if (.not. in_active) then
            k = k + 1
            call cp_fm_to_fm(tmp_fm, zmos(ispin)%matrix, 1, k, i)
          endif
        enddo
        CPASSERT(k == ncore)
        call cp_fm_release(tmp_fm)
      enddo
      call cp_fm_release(kstot2_fm)

   END SUBROUTINE get_response_zmos

! **************************************************************************************************
!> \brief Restores the qs_env to the reference
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE restore_qs_env(qs_env, active_space_env)

      TYPE(qs_environment_type), POINTER, intent(inout)  :: qs_env
      TYPE(active_space_type), POINTER, intent(in)       :: active_space_env

      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      INTEGER :: nspins, ispin

      call get_qs_env(qs_env, rho=rho, ks_env=ks_env)
      call qs_rho_get(rho, rho_ao=rho_ao)
      nspins = active_space_env%nspins
      do ispin = 1, nspins
         call dbcsr_copy(rho_ao(ispin)%matrix, active_space_env%pmat_reference(ispin)%matrix)
      enddo
      call qs_rho_update_rho(rho, qs_env)
      call qs_ks_did_change(ks_env, rho_changed=.true.)

      qs_env%requires_mo_derivs = .False.
      call qs_ks_update_qs_env(qs_env)

   END SUBROUTINE restore_qs_env

! **************************************************************************************************
!> \brief Calculates the forces related to the CASCI
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE qs_calculate_forces_active_space(qs_env)

      TYPE(qs_environment_type), POINTER, intent(inout)  :: qs_env

      TYPE(active_space_type), POINTER                   :: as_env
      TYPE(qs_force_type), POINTER, DIMENSION(:)         :: force, nuclear_force 
      TYPE(cp_para_env_type), POINTER                    :: para_env
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      TYPE(neighbor_list_set_p_type), DIMENSION(:), POINTER :: sab_orb
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(qs_energy_type), POINTER                      :: energy

      LOGICAL                                            :: requires_mo_derivs_old
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: natom_of_kind
      INTEGER :: nkind, nspins, ispin

      CPASSERT(ASSOCIATED(qs_env%active_space))
      nullify(ks_env, as_env, para_env, atomic_kind_set, dft_control, force, sab_orb)
      CALL get_qs_env(qs_env, para_env=para_env, ks_env=ks_env, active_space=as_env, &
                      dft_control=dft_control, atomic_kind_set=atomic_kind_set, force=force, &
                      sab_orb=sab_orb, energy=energy)
      requires_mo_derivs_old = qs_env%requires_mo_derivs
      qs_env%requires_mo_derivs = .false.

      nspins = as_env%nspins

      nullify(nuclear_force)
      nkind = SIZE(atomic_kind_set)
      allocate(natom_of_kind(nkind))
      call get_atomic_kind_set(atomic_kind_set=atomic_kind_set, natom_of_kind=natom_of_kind)
      call allocate_qs_force(nuclear_force, natom_of_kind)
      call zero_qs_force(nuclear_force)
      deallocate(natom_of_kind)

      nullify(rho_ao)
      CALL dbcsr_allocate_matrix_set(rho_ao, nspins)
      do ispin = 1, nspins
        allocate(rho_ao(ispin)%matrix)
        call dbcsr_copy(rho_ao(ispin)%matrix, as_env%pmat_reference(ispin)%matrix)
        call dbcsr_set(rho_ao(ispin)%matrix, 0.0_dp)
      enddo
      call set_qs_env(qs_env, force=nuclear_force)

      !! Calculate nuclear force
      call add_1rdm_force_contribution(qs_env, rho_ao, alpha=0.0_dp, beta=1.0_dp)

      call set_qs_env(qs_env, force=force)
      call sum_qs_force(force, as_env%rdm2_force, alpha=0.0_dp, beta=1.0_dp)

      !! HF x Z-matrix
      call add_1rdm_force_contribution_mixed(qs_env, as_env%pmat_reference, as_env%pmat_z, nuclear_force)

      !! Reference x Reference and nuclear contribution
      call add_1rdm_force_contribution(qs_env, as_env%pmat_reference)

      do ispin = 1, nspins
        call dbcsr_copy(rho_ao(ispin)%matrix, as_env%pmat_total(ispin)%matrix)
        call dbcsr_add(rho_ao(ispin)%matrix, as_env%pmat_reference(ispin)%matrix, 1.0_dp, -1.0_dp)
      enddo
      !! Core x Active
      call add_1rdm_force_contribution_mixed(qs_env, as_env%pmat_inactive, rho_ao, nuclear_force)

      ! Overlap contribution
      IF (nspins == 2) THEN
         CALL dbcsr_add(as_env%pmat_x(1)%matrix, as_env%pmat_x(2)%matrix, &
                        alpha_scalar=1.0_dp, beta_scalar=1.0_dp)
      END IF
      CALL dbcsr_scale(as_env%pmat_x(1)%matrix, 0.5_dp)

      CALL build_overlap_matrix(ks_env, matrix_s=rho_ao, &
                                basis_type_a="ORB", basis_type_b="ORB", &
                                sab_nl=sab_orb, calculate_forces=.TRUE., &
                                matrix_p=as_env%pmat_x(1)%matrix)

      CALL dbcsr_scale(as_env%pmat_x(1)%matrix, 2.0_dp)
      IF (nspins == 2) THEN
         CALL dbcsr_add(as_env%pmat_x(1)%matrix, as_env%pmat_x(2)%matrix, &
                        alpha_scalar=1.0_dp, beta_scalar=-1.0_dp)
      END IF

      call dbcsr_deallocate_matrix_set(rho_ao)
      call deallocate_qs_force(nuclear_force)
      call restore_qs_env(qs_env, as_env)
      energy%total = as_env%energy_total
      qs_env%requires_mo_derivs = requires_mo_derivs_old

   END SUBROUTINE qs_calculate_forces_active_space


! **************************************************************************************************
!> \brief Calculate some forcesj
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE add_1rdm_force_contribution(qs_env, rho_ao, alpha, beta)

      TYPE(qs_environment_type), POINTER, intent(inout)  :: qs_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER, intent(in) :: rho_ao
      REAL(kind=dp), OPTIONAL :: alpha, beta

      TYPE(cp_para_env_type), POINTER                    :: para_env
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(lri_environment_type), POINTER                :: lri_env
      TYPE(qs_force_type), POINTER, DIMENSION(:)         :: force, force_temp
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_old

      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: natom_of_kind
      INTEGER :: nkind, ikind
      REAL(kind=dp) :: my_alpha, my_beta

      CALL get_qs_env(qs_env, rho=rho, para_env=para_env, ks_env=ks_env, dft_control=dft_control, &
                      atomic_kind_set=atomic_kind_set, force=force)
      nullify(force_temp)
      nkind = SIZE(atomic_kind_set)
      allocate(natom_of_kind(nkind))
      CALL get_atomic_kind_set(atomic_kind_set=atomic_kind_set, natom_of_kind=natom_of_kind)
      CALL allocate_qs_force(force_temp, natom_of_kind)
      deallocate(natom_of_kind)
      call qs_rho_get(rho, rho_ao=rho_old)

      my_alpha = 1.0_dp
      my_beta = 1.0_dp
      if (present(alpha)) my_alpha = alpha
      if (present(beta)) my_beta = beta


      call qs_rho_set(rho, rho_ao=rho_ao)
      call qs_rho_update_rho(rho, qs_env)
      call qs_ks_did_change(ks_env, rho_changed=.true.)
      call zero_qs_force(force_temp)
      call set_qs_env(qs_env, force=force_temp)

      !! Calculate forces
      CALL build_core_hamiltonian_matrix(qs_env=qs_env, calculate_forces=.TRUE.)
      CALL calculate_ecore_self(qs_env)
      CALL calculate_ecore_overlap(qs_env, para_env, calculate_forces=.TRUE.)
      CALL calculate_ecore_efield(qs_env, calculate_forces=.TRUE.)
      !swap external_e_potential before external_c_potential, to ensure
      !that external potential on grid is loaded before calculating energy of cores
      CALL external_e_potential(qs_env)
      IF (.NOT. dft_control%qs_control%gapw) THEN
         CALL external_c_potential(qs_env, calculate_forces=.TRUE.)
      END IF
      ! RIGPW  matrices
      IF (dft_control%qs_control%rigpw) THEN
         CALL get_qs_env(qs_env=qs_env, lri_env=lri_env)
         CALL build_ri_matrices(lri_env, qs_env, calculate_forces=.TRUE.)
      ENDIF

      ! Compute grid-based forces
      CALL qs_ks_update_qs_env(qs_env, calculate_forces=.TRUE.)

      ! the previous steps could erronously include W matrix contributions if
      ! W matrix is set by other routines. So first set overlap contributions to zero
      do ikind = 1, nkind
        force_temp(ikind)%overlap(:, :) = 0.0_dp
      enddo

      call set_qs_env(qs_env, force=force)
      call sum_qs_force(force, force_temp, alpha=my_alpha, beta=my_beta)
      call deallocate_qs_force(force_temp)

      !! Set everything back
      call qs_rho_set(rho, rho_ao=rho_old)
      call qs_rho_update_rho(rho, qs_env)
      call qs_ks_did_change(ks_env, rho_changed=.true.)

   END SUBROUTINE add_1rdm_force_contribution


! **************************************************************************************************
!> \brief Calculate some forces
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE add_1rdm_force_contribution_mixed(qs_env, rho_a, rho_b, nuclear_force)

      TYPE(qs_environment_type), POINTER, intent(inout)  :: qs_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER, intent(in) :: rho_a
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER, intent(in) :: rho_b
      TYPE(qs_force_type), POINTER, DIMENSION(:), intent(in) :: nuclear_force

      TYPE(qs_force_type), POINTER, DIMENSION(:)         :: force
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      INTEGER :: ispin, nspins
      REAL(kind=dp), parameter :: phi = (1.0_dp + sqrt(5.0_dp)) / 2.0_dp
      REAL(kind=dp), parameter :: phi_dagger = (1.0_dp - sqrt(5.0_dp)) / 2.0_dp

                                                            
      CALL get_qs_env(qs_env, dft_control=dft_control, force=force)
      nspins = dft_control%nspins

      nullify(rho_ao)
      CALL dbcsr_allocate_matrix_set(rho_ao, nspins)
      ! rho_a x rho_a
      do ispin = 1, nspins
        allocate(rho_ao(ispin)%matrix)
        call dbcsr_copy(rho_ao(ispin)%matrix, rho_a(ispin)%matrix)
        call dbcsr_scale(rho_ao(ispin)%matrix, phi_dagger)
      enddo
      call add_1rdm_force_contribution(qs_env, rho_ao, alpha=1.0_dp, beta=1.0_dp)

      ! rho_a + rho_b x rho_a + rho_b
      do ispin = 1, nspins
        call dbcsr_add(rho_ao(ispin)%matrix, rho_b(ispin)%matrix, 1.0_dp, phi)
      enddo
      call add_1rdm_force_contribution(qs_env, rho_ao, alpha=1.0_dp, beta=-1.0_dp)

      ! rho_b x rho_b
      do ispin = 1, nspins
        call dbcsr_copy(rho_ao(ispin)%matrix, rho_b(ispin)%matrix)
      enddo
      call add_1rdm_force_contribution(qs_env, rho_ao, alpha=1.0_dp, beta=(1+phi))
      call dbcsr_deallocate_matrix_set(rho_ao)

      ! undo the contribution of nuclear force that got counted too much
      call sum_qs_force(force, nuclear_force, alpha=1.0_dp, beta=-(1+phi))

   END SUBROUTINE add_1rdm_force_contribution_mixed
! **************************************************************************************************
!> \brief computes the one-electron operators in the subspace of the provided orbital set
!> \param mos the molecular orbital set within the active subspace
!> \param qs_env ...
!> \param active_space_env ...
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE calculate_operators(mos, qs_env, active_space_env)

      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(active_space_type), POINTER                   :: active_space_env

      CHARACTER(len=*), PARAMETER :: routineN = 'calculate_operators'

      INTEGER                                            :: handle, is, nmo, nspins
      TYPE(cp_fm_type), POINTER                          :: hmat, ksmat, mo_coeff, vxcmat
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_vxc
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: h_matrix, ks_matrix

      CALL timeset(routineN, handle)

      nspins = active_space_env%nspins

      ! Kohn-Sham / Fock operator
      IF (ASSOCIATED(active_space_env%ks_sub)) THEN
         DO is = 1, SIZE(active_space_env%ks_sub)
            CALL cp_fm_release(active_space_env%ks_sub(is)%matrix)
         END DO
         DEALLOCATE (active_space_env%ks_sub)
      END IF
      CALL get_qs_env(qs_env, matrix_ks_kp=ks_matrix)
      IF (SIZE(ks_matrix, 2) > 1) THEN
         CPABORT("No k-points allowed at this point")
      END IF
      ALLOCATE (active_space_env%ks_sub(nspins))
      DO is = 1, nspins
         CALL get_mo_set(mo_set=mos(is)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
         NULLIFY (ksmat)
         CALL subspace_operator(mo_coeff, nmo, ks_matrix(is, 1)%matrix, ksmat)
         active_space_env%ks_sub(is)%matrix => ksmat
      END DO

      ! Vxc matrix
      IF (ASSOCIATED(active_space_env%vxc_sub)) THEN
         DO is = 1, SIZE(active_space_env%vxc_sub)
            CALL cp_fm_release(active_space_env%vxc_sub(is)%matrix)
         END DO
         DEALLOCATE (active_space_env%vxc_sub)
      END IF
      NULLIFY (matrix_vxc)
      CALL get_qs_env(qs_env, matrix_vxc=matrix_vxc)
      IF (ASSOCIATED(matrix_vxc)) THEN
         ALLOCATE (active_space_env%vxc_sub(nspins))
         DO is = 1, nspins
            CALL get_mo_set(mo_set=mos(is)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
            NULLIFY (vxcmat)
            CALL subspace_operator(mo_coeff, nmo, matrix_vxc(is)%matrix, vxcmat)
            active_space_env%vxc_sub(is)%matrix => vxcmat
         END DO
      END IF

      ! Core Hamiltonian
      IF (ASSOCIATED(active_space_env%h_sub)) THEN
         DO is = 1, SIZE(active_space_env%h_sub)
            CALL cp_fm_release(active_space_env%h_sub(is)%matrix)
         END DO
         DEALLOCATE (active_space_env%h_sub)
      END IF
      NULLIFY (h_matrix)
      CALL get_qs_env(qs_env=qs_env, matrix_h_kp=h_matrix)
      IF (SIZE(h_matrix, 2) > 1) THEN
         CPABORT("No k-points allowed at this point")
      END IF
      ALLOCATE (active_space_env%h_sub(nspins))
      DO is = 1, nspins
         CALL get_mo_set(mo_set=mos(is)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
         NULLIFY (hmat)
         CALL subspace_operator(mo_coeff, nmo, h_matrix(1, 1)%matrix, hmat)
         active_space_env%h_sub(is)%matrix => hmat
      END DO

      CALL timestop(handle)

   END SUBROUTINE calculate_operators

! **************************************************************************************************
!> \brief computes a one-electron operator in the subspace of the provided orbital set
!> \param orbitals the orbital coefficient matrix
!> \param nmo the number of orbitals
!> \param op_matrix operator matrix in AO basis
!> \param op_sub operator in orbital basis
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE subspace_operator(orbitals, nmo, op_matrix, op_sub)

      TYPE(cp_fm_type), POINTER                          :: orbitals
      INTEGER, INTENT(IN)                                :: nmo
      TYPE(dbcsr_type), POINTER                          :: op_matrix
      TYPE(cp_fm_type), POINTER                          :: op_sub

      CHARACTER(len=*), PARAMETER                        :: routineN = 'subspace_operator'

      INTEGER                                            :: handle, ncol, nrow
      TYPE(cp_fm_type), POINTER                          :: vectors

      CALL timeset(routineN, handle)

      CPASSERT(.NOT. ASSOCIATED(op_sub))

      CALL cp_fm_get_info(matrix=orbitals, ncol_global=ncol, nrow_global=nrow)
      CPASSERT(nmo <= ncol)

      IF (nmo > 0) THEN

         CALL cp_fm_create(vectors, orbitals%matrix_struct, "vectors")

         CALL create_subspace_matrix(orbitals, op_sub, nmo)

         CALL cp_dbcsr_sm_fm_multiply(op_matrix, orbitals, vectors, nmo)

         CALL cp_gemm('T', 'N', nmo, nmo, nrow, 1.0_dp, orbitals, vectors, 0.0_dp, op_sub)

         CALL cp_fm_release(vectors)

      END IF

      CALL timestop(handle)

   END SUBROUTINE subspace_operator

! **************************************************************************************************
!> \brief creates a matrix of subspace size
!> \param orbitals the orbital coefficient matrix
!> \param op_sub operator in orbital basis
!> \param n the number of orbitals
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE create_subspace_matrix(orbitals, op_sub, n)

      TYPE(cp_fm_type), POINTER                          :: orbitals, op_sub
      INTEGER, INTENT(IN)                                :: n

      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct

      CPASSERT(.NOT. ASSOCIATED(op_sub))

      IF (n > 0) THEN

         NULLIFY (fm_struct)
         CALL cp_fm_struct_create(fm_struct, nrow_global=n, ncol_global=n, &
                                  para_env=orbitals%matrix_struct%para_env, &
                                  context=orbitals%matrix_struct%context)
         CALL cp_fm_create(op_sub, fm_struct, name="Subspace operator")
         CALL cp_fm_struct_release(fm_struct)

      END IF

   END SUBROUTINE create_subspace_matrix

! **************************************************************************************************
!> \brief computes a electron repulsion integrals using GPW technology
!> \param mos the molecular orbital set within the active subspace
!> \param eri_env ...
!> \param qs_env ...
!> \param iw ...
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE calculate_eri_gpw(mos, eri_env, qs_env, iw, pw_env_sub)

      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos
      TYPE(eri_type)                                     :: eri_env
      TYPE(qs_environment_type), POINTER                 :: qs_env
      INTEGER, INTENT(IN)                                :: iw
      TYPE(pw_env_type), POINTER                         :: pw_env_sub

      CHARACTER(len=*), PARAMETER                        :: routineN = 'calculate_eri_gpw'

      INTEGER :: handle, icount2, intcount, irange(2), isp, isp1, isp2, ispin, iwa1, &
         iwa12, iwa2, iwb1, iwb2, iwbs, iwbt, iwfn, nmm, nmo, nmo1, nmo2, nspins, nx
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: eri_index
      LOGICAL                                            :: print1, print2, &
                                                            skip_load_balance_distributed
      REAL(KIND=dp)                                      :: dvol, erint, pair_int, rc
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: eri
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb_sub
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(pw_p_type)                                    :: pot_g, rho_g, rho_r, wfn_r
      TYPE(pw_p_type), DIMENSION(:, :), POINTER          :: wfn_a
      TYPE(pw_p_type), POINTER                           :: wfn1, wfn2, wfn3, wfn4
      TYPE(pw_poisson_type), POINTER                     :: poisson_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(task_list_type), POINTER                      :: task_list_sub

      CALL timeset(routineN, handle)

      ! print levels
      SELECT CASE (eri_env%eri_gpw%print_level)
      CASE (silent_print_level)
         print1 = .FALSE.
         print2 = .FALSE.
      CASE (low_print_level)
         print1 = .FALSE.
         print2 = .FALSE.
      CASE (medium_print_level)
         print1 = .TRUE.
         print2 = .FALSE.
      CASE (high_print_level)
         print1 = .TRUE.
         print2 = .TRUE.
      CASE (debug_print_level)
         print1 = .TRUE.
         print2 = .TRUE.
      CASE DEFAULT
         ! do nothing
      END SELECT

      CALL get_qs_env(qs_env, dft_control=dft_control)
      CALL pw_env_get(pw_env_sub, auxbas_pw_pool=auxbas_pw_pool, &
                      poisson_env=poisson_env)
      IF (eri_env%eri_gpw%redo_poisson) THEN
         IF (.NOT. (poisson_env%parameters%solver == pw_poisson_analytic .OR. &
                    poisson_env%parameters%solver == pw_poisson_periodic)) THEN
            CPABORT("Only use simple analytic Poisson solvers with ERI calculation")
         END IF
         poisson_env%parameters%periodic = eri_env%periodicity
         CALL pw_poisson_rebuild(poisson_env)
         IF (eri_env%cutoff_radius > 0.0_dp) THEN
            poisson_env%green_fft%radius = eri_env%cutoff_radius
         ELSE
            CALL get_qs_env(qs_env, cell=cell)
            rc = cell%hmat(1, 1)
            DO iwa1 = 1, 3
               rc = MIN(rc, 0.5_dp*cell%hmat(iwa1, iwa1))
            END DO
            poisson_env%green_fft%radius = rc
         END IF
         CALL pw_eri_green_create(poisson_env%green_fft, eri_env)
         IF (print1 .AND. iw > 0) THEN
            WRITE (iw, "(T4,'ERI_GPW|',' Redefine Poisson Greens function ')")
         END IF
      END IF

      IF (eri_env%method == eri_method_gpw_ht) THEN
         ! We need a task list
         NULLIFY (task_list_sub)
         skip_load_balance_distributed = dft_control%qs_control%skip_load_balance_distributed
         CALL get_qs_env(qs_env, ks_env=ks_env, sab_orb=sab_orb_sub)
         CALL allocate_task_list(task_list_sub)
         CALL generate_qs_task_list(ks_env, task_list_sub, &
                                    reorder_rs_grid_ranks=.TRUE., soft_valid=.FALSE., &
                                    skip_load_balance_distributed=skip_load_balance_distributed, &
                                    pw_env_external=pw_env_sub, sab_orb_external=sab_orb_sub)
      END IF

      NULLIFY (wfn_r%pw, rho_g%pw)
      CALL pw_pool_create_pw(auxbas_pw_pool, wfn_r%pw, use_data=REALDATA3D, in_space=REALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)
      CALL get_qs_env(qs_env, qs_kind_set=qs_kind_set, cell=cell, &
                      particle_set=particle_set, atomic_kind_set=atomic_kind_set)

      nspins = SIZE(mos)
      ! pre-calculate wavefunctions on reals space grid
      call put_orbitals_on_grid(mos, wfn_a, qs_env, pw_env_sub)

      ! get some of the grids ready
      NULLIFY (rho_r%pw, pot_g%pw)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_r%pw, use_data=REALDATA3D, in_space=REALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, pot_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)

      ! run the FFT once, to set up buffers and to take into account the memory
      rho_r%pw%cr3d = 0.0D0
      CALL pw_transfer(rho_r%pw, rho_g%pw)
      dvol = rho_r%pw%pw_grid%dvol

      ! calculate the integrals
      intcount = 0
      DO isp1 = 1, nspins
         CALL get_mo_set(mo_set=mos(isp1)%mo_set, nmo=nmo1)
         nmm = (nmo1*(nmo1 + 1))/2
         irange = get_irange_csr(nmm, eri_env%eri(1)%csr_mat%mp_group)
         DO iwa1 = 1, nmo1
            IF (eri_env%eri_gpw%store_wfn) THEN
               wfn1 => wfn_a(iwa1, isp1)
            ELSE
               CPABORT("")
            END IF
            DO iwa2 = iwa1, nmo1
               iwa12 = csr_idx_to_combined(iwa1, iwa2, nmo1)
               IF (iwa12 >= irange(1) .AND. iwa12 <= irange(2)) THEN
                  iwa12 = iwa12 - irange(1) + 1
               ELSE
                  iwa12 = 0
               END IF
               IF (eri_env%eri_gpw%store_wfn) THEN
                  wfn2 => wfn_a(iwa2, isp1)
               ELSE
                  CPABORT("")
               END IF
               ! calculate charge distribution and potential
               rho_r%pw%cr3d = wfn1%pw%cr3d*wfn2%pw%cr3d
               IF (print2 .AND. iw > 0) THEN
                  erint = pw_integrate_function(rho_r%pw)/dvol
                  WRITE (iw, "(T4,'ERI_GPW| Integral rho_ab ',T32,2I4,' [',I1,']',T58,G20.14)") &
                     iwa1, iwa2, isp1, erint
               END IF
               CALL pw_transfer(rho_r%pw, rho_g%pw)
               CALL pw_poisson_solve(poisson_env, rho_g%pw, pair_int, pot_g%pw)
               ! screening using pair_int
               IF (pair_int < eri_env%eps_integral) CYCLE
               CALL pw_transfer(pot_g%pw, rho_r%pw)
               !
               IF (eri_env%method == eri_method_gpw_ht) THEN
                  CPABORT("Not available")
               ELSEIF (eri_env%method == eri_method_full_gpw) THEN
                  DO isp2 = isp1, nspins
                     CALL get_mo_set(mo_set=mos(isp1)%mo_set, nmo=nmo2)
                     nx = (nmo2*(nmo2 + 1))/2
                     ALLOCATE (eri(nx), eri_index(nx))
                     icount2 = 0
                     iwbs = 1
                     IF (isp1 == isp2) iwbs = iwa1
                     isp = (isp1 - 1)*isp2 - ((isp1 - 1)*(isp1 - 2))/2 + (isp2 - isp1 + 1)
                     DO iwb1 = iwbs, nmo2
                        IF (eri_env%eri_gpw%store_wfn) THEN
                           wfn3 => wfn_a(iwb1, isp2)
                        ELSE
                           CPABORT("")
                        END IF
                        iwbt = iwb1
                        IF (isp1 == isp2 .AND. iwa1 == iwb1) iwbt = iwa2
                        DO iwb2 = iwbt, nmo2
                           IF (eri_env%eri_gpw%store_wfn) THEN
                              wfn4 => wfn_a(iwb2, isp2)
                           ELSE
                              CPABORT("")
                           END IF
                           wfn_r%pw%cr3d = rho_r%pw%cr3d*wfn3%pw%cr3d*wfn4%pw%cr3d
                           erint = pw_integrate_function(wfn_r%pw)
                           IF (ABS(erint) > eri_env%eps_integral) THEN
                              intcount = intcount + 1
                              IF (print2 .AND. iw > 0) THEN
                                 WRITE (iw, "(T4,'ERI_GPW|',T20,2I4,' [',I1,']',2I4,' [',I1,']',T58,G20.14)") &
                                    iwa1, iwa2, isp1, iwb1, iwb2, isp2, erint
                              END IF
                              icount2 = icount2 + 1
                              eri(icount2) = erint
                              eri_index(icount2) = csr_idx_to_combined(iwb1, iwb2, nmo2)
                           END IF
                        END DO
                     END DO
                     !
                     CALL update_csr_matrix(eri_env%eri(isp)%csr_mat, icount2, eri, eri_index, iwa12)
                     !
                     DEALLOCATE (eri, eri_index)
                  END DO
               ELSE
                  CPABORT("Unknown option")
               END IF
            END DO
         END DO
      END DO

      IF (print1 .AND. iw > 0) THEN
         WRITE (iw, "(T4,'ERI_GPW|',' Number of Integrals stored ',T68,I10)") intcount
      END IF

      IF (eri_env%eri_gpw%store_wfn) THEN
         DO ispin = 1, nspins
            CALL get_mo_set(mo_set=mos(ispin)%mo_set, nmo=nmo)
            DO iwfn = 1, nmo
               CALL pw_release(wfn_a(iwfn, ispin)%pw)
            END DO
         END DO
         DEALLOCATE (wfn_a)
      END IF
      CALL pw_pool_give_back_pw(auxbas_pw_pool, wfn_r%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_g%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_r%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, pot_g%pw)

      IF (eri_env%method == eri_method_gpw_ht) THEN
         CALL deallocate_task_list(task_list_sub)
      END IF

      CALL timestop(handle)

   END SUBROUTINE calculate_eri_gpw

! **************************************************************************************************
!> \brief Sets the Green's function
!> \param green ...
!> \param eri_env ...
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE pw_eri_green_create(green, eri_env)

      TYPE(greens_fn_type), POINTER                      :: green
      TYPE(eri_type)                                     :: eri_env

      INTEGER                                            :: ig
      REAL(KIND=dp)                                      :: a, ea, g2, g3d, ga, gg, rg, rlength
      TYPE(pw_grid_type), POINTER                        :: grid
      TYPE(pw_type), POINTER                             :: gf

      ! initialize influence function
      gf => green%influence_fn
      grid => green%influence_fn%pw_grid
      SELECT CASE (green%method)
      CASE (PERIODIC3D)

         SELECT CASE (eri_env%operator)
         CASE (eri_operator_coulomb)
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gf%cc(ig) = fourpi/g2
            END DO
            IF (grid%have_g0) gf%cc(1) = 0.0_dp
         CASE (eri_operator_yukawa)
            a = eri_env%operator_parameter**2
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gf%cc(ig) = fourpi/(a + g2)
            END DO
            IF (grid%have_g0) gf%cc(1) = fourpi/a
         CASE (eri_operator_erf)
            a = eri_env%operator_parameter**2
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               ga = -0.25_dp*g2/a
               gf%cc(ig) = fourpi/g2*EXP(ga)
            END DO
            IF (grid%have_g0) gf%cc(1) = 0.0_dp
         CASE (eri_operator_erfc)
            a = eri_env%operator_parameter**2
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               ga = -0.25_dp*g2/a
               gf%cc(ig) = fourpi/g2*(1._dp - EXP(ga))
            END DO
            IF (grid%have_g0) gf%cc(1) = 0.25_dp*fourpi/a
         CASE (eri_operator_gaussian)
            CPABORT("")
         CASE DEFAULT
            CPABORT("")
         END SELECT

      CASE (ANALYTIC0D)

         SELECT CASE (eri_env%operator)
         CASE (eri_operator_coulomb)
            rlength = green%radius
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gg = SQRT(g2)
               g3d = fourpi/g2
               gf%cc(ig) = g3d*(1.0_dp - COS(rlength*gg))
            END DO
            IF (grid%have_g0) gf%cc(1) = 0.5_dp*fourpi*rlength*rlength
         CASE (eri_operator_yukawa)
            rlength = green%radius
            a = eri_env%operator_parameter
            ea = EXP(-a*rlength)
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gg = SQRT(g2)
               g3d = fourpi/(a*a + g2)
               rg = rlength*gg
               gf%cc(ig) = g3d*(1.0_dp - ea*(COS(rg) + a/gg*SIN(rg)))
            END DO
            IF (grid%have_g0) gf%cc(1) = fourpi/(a*a)*(1.0_dp - ea*(1._dp + a*rlength))
         CASE (eri_operator_erf)
            rlength = green%radius
            a = eri_env%operator_parameter**2
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gg = SQRT(g2)
               ga = -0.25_dp*g2/a
               gf%cc(ig) = fourpi/g2*EXP(ga)*(1.0_dp - COS(rlength*gg))
            END DO
            IF (grid%have_g0) gf%cc(1) = 0.5_dp*fourpi*rlength*rlength
         CASE (eri_operator_erfc)
            rlength = green%radius
            a = eri_env%operator_parameter**2
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               g2 = grid%gsq(ig)
               gg = SQRT(g2)
               ga = -0.25_dp*g2/a
               gf%cc(ig) = fourpi/g2*(1._dp - EXP(ga))*(1.0_dp - COS(rlength*gg))
            END DO
            IF (grid%have_g0) gf%cc(1) = 0._dp
         CASE (eri_operator_gaussian)
            CPABORT("")
         CASE DEFAULT
            CPABORT("")
         END SELECT

      CASE DEFAULT
         CPABORT("")
      END SELECT

   END SUBROUTINE pw_eri_green_create

! **************************************************************************************************
!> \brief Adds data for a new row to the csr matrix
!> \param csr_mat ...
!> \param nnz ...
!> \param rdat ...
!> \param rind ...
!> \param irow ...
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE update_csr_matrix(csr_mat, nnz, rdat, rind, irow)

      TYPE(dbcsr_csr_type), INTENT(INOUT)                :: csr_mat
      INTEGER, INTENT(IN)                                :: nnz
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: rdat
      INTEGER, DIMENSION(:), INTENT(IN)                  :: rind
      INTEGER, INTENT(IN)                                :: irow

      INTEGER                                            :: k, nrow, nze, nze_new

      IF (irow /= 0) THEN
         CPASSERT(irow <= csr_mat%nrows_total)
         CPASSERT(irow > csr_mat%nrows_local)
         nze = csr_mat%nze_local
         nze_new = nze + nnz
         ! values
         CALL reallocate(csr_mat%nzval_local%r_dp, 1, nze_new)
         csr_mat%nzval_local%r_dp(nze + 1:nze_new) = rdat(1:nnz)
         ! col indices
         CALL reallocate(csr_mat%colind_local, 1, nze_new)
         csr_mat%colind_local(nze + 1:nze_new) = rind(1:nnz)
         ! rows
         nrow = csr_mat%nrows_local
         CALL reallocate(csr_mat%rowptr_local, 1, irow + 1)
         csr_mat%rowptr_local(nrow + 1:irow) = nze + 1
         csr_mat%rowptr_local(irow + 1) = nze_new + 1
         ! nzerow
         CALL reallocate(csr_mat%nzerow_local, 1, irow)
         DO k = nrow + 1, irow
            csr_mat%nzerow_local(k) = csr_mat%rowptr_local(k + 1) - csr_mat%rowptr_local(k)
         END DO
         csr_mat%nrows_local = irow
         csr_mat%nze_local = csr_mat%nze_local + nnz
      END IF
      csr_mat%nze_total = csr_mat%nze_total + nnz
      csr_mat%has_indices = .TRUE.

   END SUBROUTINE update_csr_matrix

! **************************************************************************************************
!> \brief Computes and prints the active orbitals on Cube Files
!> \param input ...
!> \param qs_env the qs_env in which the qs_env lives
!> \param mos ...
! **************************************************************************************************
   SUBROUTINE print_orbital_cubes(input, qs_env, mos)
      TYPE(section_vals_type), POINTER                   :: input
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mo_set_p_type), DIMENSION(:)                  :: mos

      CHARACTER(LEN=default_path_length)                 :: filebody, filename, title
      INTEGER                                            :: i, imo, isp, nmo, str(3), unit_nr
      INTEGER, DIMENSION(:), POINTER                     :: alist, blist, istride
      LOGICAL                                            :: do_mo, explicit_a, explicit_b
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(cp_fm_type), POINTER                          :: mo_coeff
      TYPE(cp_para_env_type), POINTER                    :: para_env
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(particle_list_type), POINTER                  :: particles
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(pw_env_type), POINTER                         :: pw_env
      TYPE(pw_p_type)                                    :: wf_g, wf_r
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_subsys_type), POINTER                      :: subsys

      CALL section_vals_val_get(input, "FILENAME", c_val=filebody)
      CALL section_vals_val_get(input, "STRIDE", i_vals=istride)
      IF (SIZE(istride) == 1) THEN
         str(1:3) = istride(1)
      ELSEIF (SIZE(istride) == 3) THEN
         str(1:3) = istride(1:3)
      ELSE
         CPABORT("STRIDE arguments inconsistent")
      END IF
      CALL section_vals_val_get(input, "ALIST", i_vals=alist, explicit=explicit_a)
      CALL section_vals_val_get(input, "BLIST", i_vals=blist, explicit=explicit_b)

      CALL get_qs_env(qs_env=qs_env, &
                      dft_control=dft_control, &
                      para_env=para_env, &
                      subsys=subsys, &
                      atomic_kind_set=atomic_kind_set, &
                      qs_kind_set=qs_kind_set, &
                      cell=cell, &
                      particle_set=particle_set, &
                      pw_env=pw_env)
      CALL qs_subsys_get(subsys, particles=particles)
      !
      CALL pw_env_get(pw_env, auxbas_pw_pool=auxbas_pw_pool)
      CALL pw_pool_create_pw(auxbas_pw_pool, wf_r%pw, use_data=REALDATA3D, in_space=REALSPACE)
      CALL pw_pool_create_pw(auxbas_pw_pool, wf_g%pw, use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)
      !
      DO isp = 1, SIZE(mos)
         CALL get_mo_set(mo_set=mos(isp)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
         DO imo = 1, nmo
            IF (isp == 1 .AND. explicit_a) THEN
               IF (alist(1) == -1) THEN
                  do_mo = .TRUE.
               ELSE
                  do_mo = .FALSE.
                  DO i = 1, SIZE(alist)
                     IF (imo == alist(i)) do_mo = .TRUE.
                  END DO
               END IF
            ELSE IF (isp == 2 .AND. explicit_b) THEN
               IF (blist(1) == -1) THEN
                  do_mo = .TRUE.
               ELSE
                  do_mo = .FALSE.
                  DO i = 1, SIZE(blist)
                     IF (imo == blist(i)) do_mo = .TRUE.
                  END DO
               END IF
            ELSE
               do_mo = .TRUE.
            END IF
            IF (.NOT. do_mo) CYCLE
            CALL calculate_wavefunction(mo_coeff, imo, wf_r, wf_g, atomic_kind_set, &
                                        qs_kind_set, cell, dft_control, particle_set, pw_env)
            IF (para_env%ionode) THEN
               WRITE (filename, '(A,A1,I4.4,A1,I1.1,A)') TRIM(filebody), "_", imo, "_", isp, ".cube"
               CALL open_file(filename, unit_number=unit_nr, file_status="UNKNOWN", file_action="WRITE")
               WRITE (title, *) "Active Orbital ", imo, " spin ", isp
            ELSE
               unit_nr = -1
            END IF
            CALL cp_pw_to_cube(wf_r%pw, unit_nr, title, particles=particles, stride=istride)
            IF (para_env%ionode) THEN
               CALL close_file(unit_nr)
            END IF
         END DO
      END DO

      CALL pw_pool_give_back_pw(auxbas_pw_pool, wf_r%pw)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, wf_g%pw)

   END SUBROUTINE print_orbital_cubes

! **************************************************************************************************
!> \brief Reads a FCIDUMP file
!> \param active_space_env ...
!> \param as_input ...
!> \par History
!>      01.2022 created [TS]
! **************************************************************************************************
   SUBROUTINE read_fcidump(active_space_env, para_env, blacs_env, iw, iw_log, rdm1_fm, rdm2_fm)
      TYPE(active_space_type), POINTER                   :: active_space_env
      TYPE(cp_para_env_type), POINTER                    :: para_env
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      type(rdm_ptr), DIMENSION(:)                        :: rdm1_fm
      type(rdm_ptr), DIMENSION(:, :, :)                  :: rdm2_fm
      integer, intent(in)                                :: iw, iw_log
      character(LEN=default_string_length)               :: line

      integer                                            :: i, j, k, l, norb, error, nmo, nao
      integer                                            :: ispin, jspin, kspin, lspin, nspins
      integer                                            :: iidx, jidx, kidx, lidx
      real(kind=dp)                                      :: element
      real(kind=dp), dimension(:,:,:), allocatable       :: rdm1
      real(kind=dp), dimension(:,:,:,:,:,:), allocatable :: rdm2
      TYPE(cp_fm_type), POINTER                          :: mo_coeff, rdm_fm, work_fm
      TYPE(cp_fm_struct_type), POINTER                   :: ao_ao_fm_struct, mo_mo_fm_struct

      nspins = active_space_env%nspins
      norb = active_space_env%eri%norb

      allocate(rdm1(norb, norb, nspins))
      allocate(rdm2(norb, norb, norb, norb, nspins, nspins))
      rdm1 = 0.0_dp
      rdm2 = 0.0_dp

      ! read the output file
      if (iw > 0) then
        do i = 1, 4
          read(iw, "(A)") line
        enddo
        do
          read(iw, '(E22.15, 4I4)', iostat=error) element, i, j, k, l
          if (error /= 0) then
            write(iw_log, *) "reached EOF"
            exit
          endif

          ispin  = (i - 1) / norb + 1
          jspin  = (j - 1) / norb + 1
          kspin  = (k - 1) / norb + 1
          lspin  = (l - 1) / norb + 1

          iidx = mod(i - 1, norb) + 1
          jidx = mod(j - 1, norb) + 1
          kidx = mod(k - 1, norb) + 1
          lidx = mod(l - 1, norb) + 1

          if (i == 0 .and. j == 0 .and. k == 0 .and. l == 0) then
            active_space_env%energy_total = element + active_space_env%energy_ref
          else if (k == 0 .and. l == 0) then
            if (ispin /= jspin) CPABORT("Error reading FCIDUMP RDM file")
            rdm1(iidx, jidx, ispin) = element
          else
            ! Not interested in these
            if (ispin /= jspin) cycle
            if (kspin /= lspin) CPABORT("Error reading FCIDUMP RDM file")
            ! notice that we are symmetrizing the first two indices
            rdm2(iidx, jidx, kidx, lidx, kspin, ispin) = rdm2(iidx, jidx, kidx, lidx, kspin, ispin) &
              + element * 0.5
            rdm2(jidx, iidx, kidx, lidx, kspin, ispin) = rdm2(jidx, iidx, kidx, lidx, kspin, ispin) &
              + element * 0.5
          endif
        enddo
      endif

      ! Set all pointers to NULL
      do ispin = 1, nspins
        nullify (rdm1_fm(ispin)%ptr)
        do i = 1, norb
          do j = 1, norb
            nullify (rdm2_fm(ispin, i, j)%ptr)
          enddo
        enddo
      enddo

      ! broadcast to all nodes
      CALL mp_bcast(active_space_env%energy_total, para_env%source, para_env%group)
      do ispin = 1, nspins
        call get_mo_set(active_space_env%mos_active(ispin)%mo_set, mo_coeff=mo_coeff, nmo=nmo, nao=nao)
        call cp_fm_struct_create(mo_mo_fm_struct, nrow_global=nmo, ncol_global=nmo, context=blacs_env)
        call cp_fm_struct_create(ao_ao_fm_struct, nrow_global=nao, ncol_global=nao, context=blacs_env)
        call cp_fm_create(work_fm, mo_coeff%matrix_struct)
        call cp_fm_create(rdm_fm, mo_mo_fm_struct)

        CALL cp_fm_create(rdm1_fm(ispin)%ptr, mo_mo_fm_struct)
        call mp_bcast(rdm1, para_env%source, para_env%group)
        call cp_fm_set_submatrix(rdm1_fm(ispin)%ptr, rdm1(:, :, ispin))

        do kspin = 1, nspins
          do i = 1, norb
            do j = i, norb
              call mp_bcast(rdm2(:, :, j, i, kspin, ispin), para_env%source, para_env%group)
              call cp_fm_set_submatrix(rdm_fm, rdm2(:, :, j, i, kspin, ispin))

              if (.not. associated(rdm2_fm(kspin, i, j)%ptr)) then
                call cp_fm_create(rdm2_fm(kspin, i, j)%ptr, ao_ao_fm_struct)
                call cp_fm_set_all(rdm2_fm(kspin, i, j)%ptr, 0.0_dp)
              endif
              ! back-transform ij-indexed 2RDM and sum over ispin
              call cp_fm_gemm('N', 'N', nao, nmo, nmo, 1.0_dp, mo_coeff, rdm_fm, 0.0_dp, work_fm)
              call cp_fm_gemm('N', 'T', nao, nao, nmo, 1.0_dp, work_fm, mo_coeff, 1.0_dp, rdm2_fm(kspin, i, j)%ptr)
            enddo
          enddo
        enddo
        call cp_fm_release(work_fm)
        call cp_fm_release(rdm_fm)
        call cp_fm_struct_release(ao_ao_fm_struct)
        call cp_fm_struct_release(mo_mo_fm_struct)
      enddo
   END SUBROUTINE read_fcidump
! **************************************************************************************************
!> \brief Writes a FCIDUMP file
!> \param active_space_env ...
!> \param as_input ...
!> \par History
!>      04.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE fcidump(active_space_env, as_input, filename)

      TYPE(active_space_type), POINTER                   :: active_space_env
      TYPE(section_vals_type), POINTER                   :: as_input

      INTEGER                                            :: i, i1, i2, i3, i4, isym, iw, nspins
      REAL(KIND=dp)                                      :: checksum, esub
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: fmat
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(eri_fcidump_checksum)                         :: eri_checksum
      CHARACTER(LEN=default_path_length), intent(out)    :: filename

      checksum = 0.0_dp

      logger => cp_get_default_logger()
      IF (BTEST(cp_print_key_should_output(logger%iter_info, &
                                           as_input, "FCIDUMP"), cp_p_file)) THEN
         iw = cp_print_key_unit_nr(logger, as_input, "FCIDUMP", &
                                   extension=".fcidump", file_status="REPLACE", file_action="WRITE", file_form="FORMATTED")
         !store the filename and return
         if (logger%para_env%ionode) then
            inquire(unit=iw, name=filename)
         endif

         nspins = active_space_env%nspins
         IF (nspins == 1) THEN
            ASSOCIATE (norb => active_space_env%eri%norb, &
                       ms2 => active_space_env%multiplicity-1, &
                       nelec => active_space_env%nelec_active)

               IF (iw > 0) THEN
                  WRITE (iw, "(A,A,I4,A,I4,A,I2,A)") "&FCI", " NORB=", norb, ",NELEC=", nelec, ",MS2=", ms2, ","
                  isym = 1
                  WRITE (iw, "(A,1000(I1,','))") "  ORBSYM=", (isym, i=1, norb)
                  isym = 0
                  WRITE (iw, "(A,I1,A)") "  ISYM=", isym, ","
                  WRITE (iw, "(A)") " /"
               END IF
               !
               ! Print integrals: ERI
               CALL active_space_env%eri%eri_foreach(1, eri_fcidump_print(iw, 1, 1))
               CALL eri_checksum%set(1, 1)
               CALL active_space_env%eri%eri_foreach(1, eri_checksum)

               ! Print integrals: Fij
               ! replicate Fock matrix
               ALLOCATE (fmat(norb, norb))
               CALL replicate_and_symmetrize_matrix(norb, active_space_env%ks_sub(1)%matrix, fmat)
               IF (iw > 0) THEN
                  i3 = 0; i4 = 0
                  DO i1 = 1, norb
                     DO i2 = i1, norb
                        checksum = checksum + ABS(fmat(i1, i2))
                        WRITE (iw, "(ES23.16,4I4)") fmat(i1, i2), i1, i2, i3, i4
                     END DO
                  END DO
               END IF
               DEALLOCATE (fmat)
               ! Print energy
               esub = active_space_env%energy_inactive
               i1 = 0; i2 = 0; i3 = 0; i4 = 0
               checksum = checksum + ABS(esub)
               IF (iw > 0) WRITE (iw, "(ES23.16,4I4)") esub, i1, i2, i3, i4
            END ASSOCIATE
         ELSE

            ASSOCIATE (norb => active_space_env%eri%norb, &
                       ms2 => active_space_env%multiplicity-1, &
                       nelec => active_space_env%nelec_active)

               IF (iw > 0) THEN
                  WRITE (iw, "(A,A,I4,A,I4,A,I2,A)") "&FCI", " NORB=", norb, ",NELEC=", nelec, ",MS2=", ms2, ","
                  isym = 1
                  WRITE (iw, "(A,1000(I1,','))") "  ORBSYM=", (isym, i=1, norb)
                  isym = 0
                  WRITE (iw, "(A,I1,A)") "  ISYM=", isym, ","
                  WRITE (iw, "(A,I1,A)") "  UHF=", 1, ","
                  WRITE (iw, "(A)") " /"
               END IF
               !
               ! Print integrals: ERI
               ! alpha-alpha
               CALL active_space_env%eri%eri_foreach(1, eri_fcidump_print(iw, 1, 1))
               CALL eri_checksum%set(1, 1)
               CALL active_space_env%eri%eri_foreach(1, eri_checksum)
               ! alpha-beta
               CALL active_space_env%eri%eri_foreach(2, eri_fcidump_print(iw, 1, norb + 1))
               CALL eri_checksum%set(1, norb + 1)
               CALL active_space_env%eri%eri_foreach(2, eri_checksum)
               ! beta-beta
               CALL active_space_env%eri%eri_foreach(3, eri_fcidump_print(iw, norb + 1, norb + 1))
               CALL eri_checksum%set(norb + 1, norb + 1)
               CALL active_space_env%eri%eri_foreach(3, eri_checksum)
               ! Print integrals: Fij
               ! alpha
               ALLOCATE (fmat(norb, norb))
               CALL replicate_and_symmetrize_matrix(norb, active_space_env%ks_sub(1)%matrix, fmat)
               IF (iw > 0) THEN
                  i3 = 0; i4 = 0
                  DO i1 = 1, norb
                     DO i2 = i1, norb
                        checksum = checksum + ABS(fmat(i1, i2))
                        WRITE (iw, "(ES23.16,4I4)") fmat(i1, i2), i1, i2, i3, i4
                     END DO
                  END DO
               END IF
               DEALLOCATE (fmat)
               ! beta
               ALLOCATE (fmat(norb, norb))
               CALL replicate_and_symmetrize_matrix(norb, active_space_env%ks_sub(2)%matrix, fmat)
               IF (iw > 0) THEN
                  i3 = 0; i4 = 0
                  DO i1 = 1, norb
                     DO i2 = i1, norb
                        checksum = checksum + ABS(fmat(i1, i2))
                        WRITE (iw, "(ES23.16,4I4)") fmat(i1, i2), i1 + norb, i2 + norb, i3, i4
                     END DO
                  END DO
               END IF
               DEALLOCATE (fmat)
               ! Print energy
               esub = active_space_env%energy_inactive
               i1 = 0; i2 = 0; i3 = 0; i4 = 0
               checksum = checksum + ABS(esub)
               IF (iw > 0) WRITE (iw, "(ES23.16,4I4)") esub, i1, i2, i3, i4
            END ASSOCIATE
         END IF
         !
         CALL cp_print_key_finished_output(iw, logger, as_input, "FCIDUMP")

         !>>
         iw = cp_logger_get_default_io_unit(logger)
         IF (iw > 0) WRITE (iw, '(T4,A,T66,F12.8)') "FCIDUMP| Checksum:", eri_checksum%checksum + checksum
         !<<

      END IF

   END SUBROUTINE fcidump

! **************************************************************************************************
!> \brief replicate and symmetrize a matrix
!> \param norb the number of orbitals
!> \param distributed_matrix ...
!> \param replicated_matrix ...
! **************************************************************************************************
   SUBROUTINE replicate_and_symmetrize_matrix(norb, distributed_matrix, replicated_matrix)
      INTEGER, INTENT(IN)                                :: norb
      TYPE(cp_fm_type), INTENT(IN), POINTER              :: distributed_matrix
      REAL(dp), DIMENSION(:, :), INTENT(INOUT)           :: replicated_matrix

      INTEGER                                            :: i1, i2
      REAL(dp)                                           :: mval

      replicated_matrix(:, :) = 0.0_dp
      DO i1 = 1, norb
         DO i2 = i1, norb
            CALL cp_fm_get_element(distributed_matrix, i1, i2, mval)
            replicated_matrix(i1, i2) = mval
            replicated_matrix(i2, i1) = mval
         END DO
      END DO
   END SUBROUTINE replicate_and_symmetrize_matrix

! **************************************************************************************************
!> \brief Creates a local basis
!> \param pro_basis_set ...
!> \param zval ...
!> \param ishell ...
!> \param nshell ...
!> \param lnam ...
!> \par History
!>      05.2016 created [JGH]
! **************************************************************************************************
   SUBROUTINE create_pro_basis(pro_basis_set, zval, ishell, nshell, lnam)
      TYPE(gto_basis_set_type), POINTER                  :: pro_basis_set
      INTEGER, INTENT(IN)                                :: zval, ishell
      INTEGER, DIMENSION(:), INTENT(IN)                  :: nshell
      CHARACTER(len=*), DIMENSION(:), INTENT(IN)         :: lnam

      CHARACTER(len=6), DIMENSION(:), POINTER            :: sym
      INTEGER                                            :: i, l, nj
      INTEGER, DIMENSION(4, 7)                           :: ne
      INTEGER, DIMENSION(:), POINTER                     :: lq, nq
      REAL(KIND=dp), DIMENSION(:), POINTER               :: zet
      TYPE(sto_basis_set_type), POINTER                  :: sto_basis_set

      CPASSERT(.NOT. ASSOCIATED(pro_basis_set))
      NULLIFY (sto_basis_set)

      ! electronic configuration
      ne = 0
      DO l = 1, 4 !lq(1)+1
         nj = 2*(l - 1) + 1
         DO i = l, 7 ! nq(1)
            ne(l, i) = ptable(zval)%e_conv(l - 1) - 2*nj*(i - l)
            ne(l, i) = MAX(ne(l, i), 0)
            ne(l, i) = MIN(ne(l, i), 2*nj)
         END DO
      END DO
      ALLOCATE (nq(ishell), lq(ishell), zet(ishell), sym(ishell))
      DO i = 1, ishell
         nq(i) = nshell(i)
         SELECT CASE (lnam(i))
         CASE ('S', 's')
            lq(i) = 0
         CASE ('P', 'p')
            lq(i) = 1
         CASE ('D', 'd')
            lq(i) = 2
         CASE ('F', 'f')
            lq(i) = 3
         CASE DEFAULT
            CPABORT("Wrong l QN")
         END SELECT
         sym(i) = lnam(i)
         zet(i) = srules(zval, ne, nq(1), lq(1))
      END DO
      CALL allocate_sto_basis_set(sto_basis_set)
      CALL set_sto_basis_set(sto_basis_set, nshell=1, nq=nq, lq=lq, zet=zet, symbol=sym)
      CALL create_gto_from_sto_basis(sto_basis_set, pro_basis_set, 6)
      pro_basis_set%norm_type = 2
      CALL init_orb_basis_set(pro_basis_set)
      CALL deallocate_sto_basis_set(sto_basis_set)

   END SUBROUTINE create_pro_basis

! **************************************************************************************************
!> \brief ...
!> the results are stored in qs_env%matrix_ks
! **************************************************************************************************
   subroutine calculate_fock(qs_env, den)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: den
      integer                                            :: ispin, nspins
      logical                                            :: requires_matrix_vxc_old, &
                                                            requires_mo_derivs_old
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dft_control_type), POINTER                    :: dft_control

      CALL get_qs_env(qs_env, rho=rho, ks_env=ks_env, dft_control=dft_control)
      CALL qs_rho_get(rho, rho_ao=rho_ao)
      nspins = dft_control%nspins
      do ispin = 1, nspins
         CALL dbcsr_copy(rho_ao(ispin)%matrix, den(ispin)%matrix)
      enddo
      call qs_rho_update_rho(rho, qs_env)
      call qs_ks_did_change(ks_env, rho_changed=.true.)

      !TODO when KS is used, this should be Hartree--Fock
      ! let's also calculate Vxc
      requires_matrix_vxc_old = qs_env%requires_matrix_vxc
      qs_env%requires_matrix_vxc = .False.

      ! for active space construction MO derivatives are not necessary (crashes otherwise)
      requires_mo_derivs_old = qs_env%requires_mo_derivs
      qs_env%requires_mo_derivs = .False.

      ! new density and Kohn-Sham matrix
      CALL qs_ks_update_qs_env(qs_env)

      ! set back
      qs_env%requires_matrix_vxc = requires_matrix_vxc_old
      qs_env%requires_mo_derivs = requires_mo_derivs_old
   end subroutine calculate_fock

! **************************************************************************************************
!> \brief Get orbitals on a grid
! **************************************************************************************************

   SUBROUTINE put_orbitals_on_grid(mos, wfn_a, qs_env, pw_env_sub)
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos
      TYPE(pw_p_type), DIMENSION(:, :), POINTER          :: wfn_a
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_env_type), POINTER                         :: pw_env_sub

      CHARACTER(len=*), PARAMETER                        :: routineN = 'put_orbitals_on_grid'

      INTEGER :: nmo, nspins, ispin, iwfn, nx

      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(cp_fm_type), POINTER                          :: mo_coeff
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(pw_p_type)                                    :: rho_g

      CALL get_qs_env(qs_env, qs_kind_set=qs_kind_set, cell=cell, &
                      particle_set=particle_set, atomic_kind_set=atomic_kind_set, &
                      dft_control=dft_control)
      CALL pw_env_get(pw_env_sub, auxbas_pw_pool=auxbas_pw_pool)

      nspins = SIZE(mos)
      nmo = 0
      DO ispin = 1, nspins
         CALL get_mo_set(mo_set=mos(ispin)%mo_set, nmo=nx)
         nmo = MAX(nmo, nx)
      END DO
      ALLOCATE (wfn_a(nmo, nspins))

      NULLIFY(rho_g%pw)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_g%pw,&
        use_data=COMPLEXDATA1D, in_space=RECIPROCALSPACE)

      DO ispin = 1, nspins
         CALL get_mo_set(mo_set=mos(ispin)%mo_set, mo_coeff=mo_coeff, nmo=nmo)
         DO iwfn = 1, nmo
            NULLIFY (wfn_a(iwfn, ispin)%pw)
            CALL pw_pool_create_pw(auxbas_pw_pool, wfn_a(iwfn, ispin)%pw, &
                                   use_data=REALDATA3D, in_space=REALSPACE)
            CALL calculate_wavefunction(mo_coeff, iwfn, wfn_a(iwfn, ispin), rho_g, atomic_kind_set, &
                                        qs_kind_set, cell, dft_control, particle_set, pw_env_sub)
         END DO
      END DO
      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_g%pw)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Overlap of current MO coefficients with others already mapped to the grid
! **************************************************************************************************

   SUBROUTINE orbital_overlaps(mos, unoccupied_orbs, wfn_a, qs_env, pw_env, active_orb_list, logger)
      TYPE(mo_set_p_type), DIMENSION(:), POINTER, intent(in) :: mos
      TYPE(pw_p_type), DIMENSION(:, :), POINTER, intent(in)  :: wfn_a
      TYPE(qs_environment_type), POINTER, intent(in)         :: qs_env
      TYPE(pw_env_type), POINTER, intent(in)                 :: pw_env
      INTEGER, DIMENSION(:, :), POINTER, intent(inout)       :: active_orb_list
      TYPE(cp_logger_type), POINTER, intent(in)              :: logger

      CHARACTER(len=*), PARAMETER                        :: routineN = 'orbital_overlaps'
      TYPE(pw_p_type), DIMENSION(:, :), POINTER          :: wfn_b
      TYPE(pw_p_type), POINTER                           :: wfn1, wfn2
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_p_type)                                    :: rho_r
      TYPE(cp_fm_p_type), DIMENSION(:), POINTER          :: unoccupied_orbs
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos_full
      TYPE(cp_fm_type), POINTER                          :: mo_coeff1, mo_coeff2
      TYPE(cp_fm_pool_type), POINTER                     :: wfn_fm_pool
      TYPE(cp_fm_struct_type), POINTER                   :: wfn_fm_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: overlaps
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: indexes

      INTEGER :: isp, iwa, iwb, nspins, nmo1, nmo2, nvirt, nao, nel, iw
      REAL(KIND=dp) :: erint, maxocc, ovlap
      iw = cp_logger_get_default_io_unit(logger)

      call get_qs_env(qs_env, blacs_env=blacs_env)

      nspins = SIZE(mos)
      nullify(mos_full)
      allocate(mos_full(nspins))
      DO isp = 1, nspins
        nel = qs_env%active_space%nelec_inactive_spinwise(isp)
        nvirt = unoccupied_orbs(isp)%matrix%matrix_struct%ncol_global

        CALL get_mo_set(mo_set=mos(isp)%mo_set, mo_coeff=mo_coeff1,&
                        maxocc=maxocc, nao=nao, nmo=nmo1)

        nullify(mos_full(isp)%mo_set)
        CALL cp_fm_struct_create(wfn_fm_struct, nrow_global=nao, ncol_global=nmo1+nvirt,&
                                 context=blacs_env)
        CALL fm_pool_create(wfn_fm_pool, wfn_fm_struct)

        CALL allocate_mo_set(mos_full(isp)%mo_set, nao, nmo1+nvirt, nel, REAL(nel, dp), &
                             maxocc, flexible_electron_count=0.0_dp)
        CALL init_mo_set(mos_full(isp)%mo_set, fm_pool=wfn_fm_pool, name="full MO")
        CALL fm_pool_release(wfn_fm_pool)
        CALL cp_fm_struct_release(wfn_fm_struct)

        CALL get_mo_set(mo_set=mos_full(isp)%mo_set, mo_coeff=mo_coeff2)

        CALL cp_fm_to_fm(mo_coeff1, mo_coeff2, nmo1, 1, 1)
        CALL cp_fm_to_fm(unoccupied_orbs(isp)%matrix, mo_coeff2, nvirt, 1, nmo1+1)
      END DO

      CALL put_orbitals_on_grid(mos_full, wfn_b, qs_env, pw_env)
      CALL pw_env_get(pw_env, auxbas_pw_pool=auxbas_pw_pool)
      CALL pw_pool_create_pw(auxbas_pw_pool, rho_r%pw, use_data=REALDATA3D, in_space=REALSPACE)



      ! run the FFT once, to set up buffers and to take into account the memory
      rho_r%pw%cr3d = 0.0D0

      DO isp = 1, nspins
         CALL get_mo_set(mo_set=qs_env%active_space%mos_active(isp)%mo_set, nmo=nmo1)
         CALL get_mo_set(mo_set=mos_full(isp)%mo_set, nmo=nmo2)
         ALLOCATE(overlaps(nmo2), indexes(nmo2))
         DO iwb = 1, nmo2
            wfn2 => wfn_b(iwb, isp)
            ovlap = 0.0
            DO iwa = 1, nmo1
               wfn1 => wfn_a(iwa, isp)
               rho_r%pw%cr3d = wfn1%pw%cr3d*wfn2%pw%cr3d
               erint = pw_integrate_function(rho_r%pw)
               ovlap = ovlap + erint * erint
               ! WRITE (iw, "(T4,'ORBITAL_OVERLAPS| orbital overlap ',T39,2I4,' [',I1,']',T58,G20.14)") iwa, iwb, isp, erint
            END DO

            overlaps(iwb) = ovlap
            WRITE (iw, "(T4,'ORBITAL_OVERLAPS| subspace overlap ',T39,I4,' [',I1,']',T58,G20.14)") iwb, isp, ovlap
         END DO
         call argsort(overlaps, indexes)
         active_orb_list(1:nmo1, isp) = indexes(1:nmo1)
         ovlap = 0
         DO iwb = 1, nmo1
          ovlap = ovlap + overlaps(indexes(iwb))
         ENDDO
         WRITE (iw, "(T4,'ORBITAL_OVERLAPS| active space overlap ',T43,G20.14)") ovlap / nmo1
         DEALLOCATE(overlaps, indexes)
      END DO
      DO isp = 1, nspins
         CALL get_mo_set(mo_set=mos_full(isp)%mo_set, nmo=nmo1)
         DO iwb = 1, nmo1
            CALL pw_release(wfn_b(iwb, isp)%pw)
         END DO
         call deallocate_mo_set(mos_full(isp)%mo_set)
      END DO
      DEALLOCATE (wfn_b, mos_full)

      CALL pw_pool_give_back_pw(auxbas_pw_pool, rho_r%pw)


   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param active_space_env ...
! **************************************************************************************************
   SUBROUTINE update_active_space(qs_env, active_space_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(active_space_type), POINTER                   :: active_space_env

      CHARACTER(len=*), PARAMETER :: routineN = 'update_active_space'

      INTEGER                                            :: handle
      TYPE(mo_set_p_type), DIMENSION(:), POINTER         :: mos
      TYPE(qs_energy_type), POINTER                      :: energy

      CALL timeset(routineN, handle)

      ! density matrix
      call calculate_fock(qs_env, active_space_env%pmat_inactive)

      ! reference energy
      CALL get_qs_env(qs_env, energy=energy)
      active_space_env%energy_inactive = energy%total

      ! update operators
      mos => active_space_env%mos_active
      CALL calculate_operators(mos, qs_env, active_space_env)

      CALL timestop(handle)
   END SUBROUTINE update_active_space

! **************************************************************************************************
!> \brief Print each value on the master node
!> \param this object reference
!> \param i i-index
!> \param j j-index
!> \param k k-index
!> \param l l-index
!> \param val value of the integral at (i,j,k.l)
!> \return always true to dump all integrals
! **************************************************************************************************
   LOGICAL FUNCTION eri_fcidump_print_func(this, i, j, k, l, val) RESULT(cont)
      CLASS(eri_fcidump_print), INTENT(inout) :: this
      INTEGER, INTENT(in)                     :: i, j, k, l
      REAL(KIND=dp), INTENT(in)               :: val

      ! write to the actual file only on the master
      IF (this%unit_nr > 0) THEN
         WRITE (this%unit_nr, "(ES23.16,4I4)") val, i + this%bra_start - 1, j + this%bra_start - 1, &
              &                                     k + this%ket_start - 1, l + this%ket_start - 1
      END IF

      cont = .TRUE.
   END FUNCTION eri_fcidump_print_func

! **************************************************************************************************
!> \brief checksum each value on the master node
!> \param this object reference
!> \param i i-index
!> \param j j-index
!> \param k k-index
!> \param l l-index
!> \param val value of the integral at (i,j,k.l)
!> \return always true to dump all integrals
! **************************************************************************************************
   LOGICAL FUNCTION eri_fcidump_checksum_func(this, i, j, k, l, val) RESULT(cont)
      CLASS(eri_fcidump_checksum), INTENT(inout) :: this
      INTEGER, INTENT(in)                     :: i, j, k, l
      REAL(KIND=dp), INTENT(in)               :: val
      MARK_USED(i)
      MARK_USED(j)
      MARK_USED(k)
      MARK_USED(l)

      this%checksum = this%checksum + ABS(val)

      cont = .TRUE.
   END FUNCTION eri_fcidump_checksum_func

   SUBROUTINE argsort(r,d)
     real(kind=dp), intent(in), dimension(:) :: r
     integer, intent(out), dimension(size(r)) :: d

     integer, dimension(size(r)) :: il

     integer :: stepsize
     integer :: i, j, n, left, k, ksize

     n = size(r)

     do i=1,n
        d(i)=i
     end do

     if (n == 1) return

     stepsize = 1
     do while (stepsize<n)
     do left=1,n-stepsize,stepsize*2
     i = left
     j = left+stepsize
     ksize = min(stepsize*2,n-left+1)
     k=1

     do while (i<left+stepsize .and. j<left+ksize)
      if (r(d(i))>r(d(j))) then
        il(k)=d(i)
        i=i+1
        k=k+1
      else
        il(k)=d(j)
        j=j+1
        k=k+1
      endif
     enddo

     if (i<left+stepsize) then
       il(k:ksize) = d(i:left+stepsize-1)
     else
       il(k:ksize) = d(j:left+ksize-1)
     endif
     d(left:left+ksize-1) = il(1:ksize)
     end do
     stepsize=stepsize*2
     end do

     return
   END SUBROUTINE

  recursive subroutine IntegerQSort(array)
    integer :: array(:), x, t
    integer :: first, last
    integer :: i, j

    first = 1
    last = size(array, 1)

    x = array( (first+last) / 2 )
    i = first
    j = last
    do
       do while (array(i) < x)
          i = i + 1
       end do
       do while (x < array(j))
          j = j - 1
       end do
       if (i >= j) exit
       t = array(i)
       array(i) = array(j)
       array(j) = t
       i = i + 1
       j = j - 1
    end do
    if (first < i-1) call IntegerQSort(array(first : i - 1))
    if (j+1 < last)  call IntegerQSort(array(j + 1: last))
  end subroutine IntegerQSort

END MODULE qs_active_space_methods
