module qs_active_space_linear
  use kinds, only: dp
  use cp_fm_types, only: cp_fm_p_type, cp_fm_to_fm, cp_fm_p_type,&
                         cp_fm_release
  use cp_fm_basic_linalg, only: cp_fm_scale, cp_fm_scale_and_add,&
                                cp_fm_trace
#include "./base/base_uses.f90"

implicit none
  private
    type(cp_fm_p_type) :: rhs

    type(cp_fm_p_type), dimension(:), allocatable :: t
    type(cp_fm_p_type), dimension(:), allocatable :: r

    real(kind=dp), dimension(:, :), allocatable :: s
    real(kind=dp), dimension(:, :), allocatable :: a
    real(kind=dp), dimension(:), allocatable :: prod

    integer :: current_size, max_size

  public :: initialize_subspace_linear
  public :: update_subspace_linear
  public :: solution_subspace_linear
  public :: terminate_subspace_linear

contains
  function my_dot_product(a, b) result(acc)
    implicit none
    type(cp_fm_p_type) :: a, b
    real(kind=dp) :: acc
    call cp_fm_trace(a%matrix, b%matrix, acc)
  end function

  subroutine my_scale(fac, a)
    implicit none
    type(cp_fm_p_type) :: a
    real(kind=dp) :: fac
    call cp_fm_scale(fac, a%matrix)
  end subroutine my_scale

  subroutine my_axpy(fac, a, b)
    implicit none
    type(cp_fm_p_type) :: a, b
    real(kind=dp) :: fac
    call cp_fm_scale_and_add(1.0d0, b%matrix, fac, a%matrix)
  end subroutine my_axpy

  subroutine my_copy(a, b)
    implicit none
    type(cp_fm_p_type) :: a, b
    call cp_fm_to_fm(a%matrix, b%matrix)
  end subroutine my_copy

  subroutine my_deallocate(a)
    implicit none
    type(cp_fm_p_type) :: a
    call cp_fm_release(a%matrix)
  end subroutine

  subroutine initialize_subspace_linear(maxsize, rin)
    implicit none
    integer :: maxsize
    type(cp_fm_p_type), intent(in) :: rin

    CPASSERT(.not. allocated(t))
    CPASSERT(.not. allocated(r))
    allocate(t(maxsize))
    allocate(r(maxsize))

    CPASSERT(.not. allocated(s))
    CPASSERT(.not. allocated(a))
    CPASSERT(.not. allocated(prod))
    allocate(prod(maxsize))
    allocate(s(maxsize, maxsize))
    allocate(a(maxsize, maxsize))

    current_size = 0
    max_size = maxsize
    rhs = rin
  end subroutine

  subroutine update_subspace_linear(tnew, rnew, residual)
    implicit none
    type(cp_fm_p_type), intent(in) :: tnew
    type(cp_fm_p_type), intent(in) :: rnew
    type(cp_fm_p_type), intent(inout) :: residual

    integer :: i, j, cnt, info, lwork
    integer, dimension(:), allocatable :: ipiv
    real(kind=dp), dimension(:), allocatable :: work

    real(kind=dp), dimension(:), allocatable :: ptmp, qtmp
    real(kind=dp), dimension(:, :), allocatable :: stmp, atmp, vtmp

    if (current_size .eq. max_size) then
      call my_deallocate(t(2))
      call my_deallocate(r(2))
      do i = 2, max_size - 1
        t(i) = t(i + 1)
        r(i) = r(i + 1)
        do j = 2, max_size - 1
          s(j, i) = s(j + 1, i + 1)
          a(j, i) = a(j + 1, i + 1)
        enddo
        s(i, 1) = s(i+1, 1)
        s(1, i) = s(1, i+1)
        a(i, 1) = a(i+1, 1)
        a(1, i) = a(1, i+1)
        prod(i) = prod(i+1)
      enddo
    else
      current_size = current_size + 1
    endif
    t(current_size) = tnew
    r(current_size) = rnew

    do i = 1, current_size - 1
      s(current_size, i) = my_dot_product(tnew, t(i))
      s(i, current_size) = s(current_size, i)
      a(current_size, i) = my_dot_product(rnew, r(i))
      a(i, current_size) = a(current_size, i)
    enddo
    s(current_size, current_size) = my_dot_product(tnew, tnew)
    a(current_size, current_size) = my_dot_product(rnew, rnew)
    prod(current_size) = - my_dot_product(rhs, rnew)

    ! solve subspace problem
    allocate(stmp(current_size, current_size))
    allocate(atmp(current_size, current_size))
    allocate(vtmp(current_size, current_size))
    allocate(ptmp(current_size))
    allocate(qtmp(current_size))
    stmp(:, :) = s(1:current_size, 1:current_size)
    atmp(:, :) = a(1:current_size, 1:current_size)
    ptmp(:) = prod(1:current_size)

    lwork = current_size * max(current_size, 10)
    allocate(work(lwork))
    call dsyev('V', 'U', current_size, stmp, current_size, qtmp, work, lwork, info)

    cnt = 0
    do i = 1, current_size
      if (qtmp(i) > 1.0d-8) then
        cnt = cnt + 1
        vtmp(:, cnt) = stmp(:, i) / sqrt(qtmp(i))
      endif
    enddo
    call dgemm('N', 'N', current_size, cnt, current_size, 1.0d0, atmp, current_size, &
               vtmp, current_size, 0.0d0, stmp, current_size)
    call dgemm('T', 'N', cnt, cnt, current_size, 1.0d0, vtmp, current_size, &
               stmp, current_size, 0.0d0, atmp, current_size)

    call dgemv('T', current_size, cnt, 1.0d0, vtmp, current_size, ptmp, 1, &
               0.0d0, qtmp, 1)

    allocate(ipiv(cnt))
    call dsysv('U', cnt, 1, atmp, cnt, ipiv, qtmp, cnt, work, lwork, info)

    call dgemv('N', current_size, cnt, 1.0d0, vtmp, current_size, qtmp, 1, &
               0.0d0, ptmp, 1)

    ! construct the residuals
    call my_copy(rhs, residual)
    do i = 1, current_size
      call my_axpy(ptmp(i), r(i), residual)
    enddo

    ! before exiting, store the best vector in no. 1
    call my_scale(ptmp(1), t(1))
    call my_scale(ptmp(1), r(1))
    do i = 2, current_size
      call my_axpy(ptmp(i), t(i), t(1))
      call my_axpy(ptmp(i), r(i), r(1))
    enddo
    stmp(:, :) = 0.0d0
    stmp(:, 1) = ptmp(:)
    do i = 2, current_size
      stmp(i, i) = 1.0d0
    enddo
    call dgemm('N', 'N', current_size, current_size, current_size, 1.0d0, s, max_size, &
               stmp, current_size, 0.0d0, atmp, current_size)
    call dgemm('T', 'N', current_size, current_size, current_size, 1.0d0, stmp, current_size, &
               atmp, current_size, 0.0d0, s, max_size)
    call dgemm('N', 'N', current_size, current_size, current_size, 1.0d0, a, max_size, &
               stmp, current_size, 0.0d0, atmp, current_size)
    call dgemm('T', 'N', current_size, current_size, current_size, 1.0d0, stmp, current_size, &
               atmp, current_size, 0.0d0, a, max_size)
    call dgemv('T', current_size, current_size, 1.0d0, stmp, current_size, prod, 1, &
               0.0d0, ptmp, 1)
    prod(1:current_size) = ptmp(1:current_size)
  end subroutine update_subspace_linear

  subroutine solution_subspace_linear(sol)
    implicit none
    type(cp_fm_p_type), intent(inout) :: sol
    call my_copy(t(1), sol)
  end subroutine solution_subspace_linear

  subroutine terminate_subspace_linear
    implicit none
    integer :: i

    do i = 1, current_size
      call my_deallocate(t(i))
      call my_deallocate(r(i))
    enddo
    call my_deallocate(rhs)

    CPASSERT(allocated(t))
    CPASSERT(allocated(r))
    deallocate(t)
    deallocate(r)

    CPASSERT(allocated(s))
    CPASSERT(allocated(a))
    CPASSERT(allocated(prod))
    deallocate(s)
    deallocate(a)
    deallocate(prod)
  end subroutine terminate_subspace_linear

end module qs_active_space_linear
