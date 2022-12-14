!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2021 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Module with functions to handle derivative descriptors.
!>      derivative description are strings have the following form
!>      "(rho)(rho)(rhoa)" which means that it is a forth order
!>      derivative, twice with respect to rho, once with respect to rhoa
!>      and once with respect to drhoa.
!>      Possible derivatives are:
!>      - rho: total density
!>      - norm_drho: norm of the gradient of the total density
!>      - rhoa, rhob: alpha and beta spin density (with LSD)
!>      - norm_drhoa, norm_drhob: norm of the gradient of the alpha and beta
!>        spin density
!>      - drhoa_drhob: scalar product between the gradient of rhoa and the
!>        grandient of rhob
!>      - tau: the local kinetic part
!>      - taua, taub: the kinetic part of the different spins
!> \note
!>       add drhox, drhoy, drhoz, drhoax,...?
!> \author thomas & fawzi
! **************************************************************************************************
MODULE xc_derivative_desc

#include "../base/base_uses.f90"
   IMPLICIT NONE

   PRIVATE

   INTEGER, PARAMETER :: &
      MAX_LABEL_LENGTH = 12, &
      MAX_DERIVATIVE = 4, &
      MAX_DERIVATIVE_DESC_LENGTH = (MAX_LABEL_LENGTH + 2)*MAX_DERIVATIVE

   LOGICAL, PARAMETER :: debug_this_module = .FALSE.

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_derivative_desc'

   PUBLIC :: MAX_DERIVATIVE_DESC_LENGTH, MAX_LABEL_LENGTH, &
             standardize_derivative_desc, &
             create_split_derivative_desc

CONTAINS

! **************************************************************************************************
!> \brief returns a unique string for a given derivative string
!> \param deriv_desc input string which describes the derivative and
!>                    should be standardized
!> \param res ...
! **************************************************************************************************
   PURE SUBROUTINE standardize_derivative_desc(deriv_desc, res)
      CHARACTER(len=*), INTENT(in)                       :: deriv_desc
      CHARACTER(LEN=MAX_DERIVATIVE_DESC_LENGTH), &
         INTENT(OUT)                                     :: res

      CHARACTER(len=MAX_LABEL_LENGTH)                    :: tmp
      CHARACTER(len=MAX_LABEL_LENGTH), DIMENSION(:), &
         POINTER                                         :: deriv_array
      INTEGER                                            :: i, l_label, pos
      LOGICAL                                            :: ordered

      CALL create_split_derivative_desc(deriv_desc, deriv_array)

      ordered = .FALSE.
      DO WHILE (.NOT. ordered)
         ordered = .TRUE.
         DO i = 1, SIZE(deriv_array) - 1
            IF (deriv_array(i) > deriv_array(i + 1)) THEN
               tmp = deriv_array(i + 1)
               deriv_array(i + 1) = deriv_array(i)
               deriv_array(i) = tmp
               ordered = .FALSE.
            END IF
         END DO
      END DO

      res = ""
      pos = 1
      DO i = 1, SIZE(deriv_array)
         l_label = LEN_TRIM(deriv_array(i))
         res(pos:pos + l_label + 1) = '('//deriv_array(i) (1:l_label)//')'
         pos = pos + l_label + 2
      END DO

      DEALLOCATE (deriv_array)

   END SUBROUTINE standardize_derivative_desc

! **************************************************************************************************
!> \brief returns an array of 1st order derivative descriptions
!> \param deriv_desc input string which describes the derivative
!> \param res ...
! **************************************************************************************************
   PURE SUBROUTINE create_split_derivative_desc(deriv_desc, res)
      CHARACTER(len=*), INTENT(in)                       :: deriv_desc
      CHARACTER(len=MAX_LABEL_LENGTH), DIMENSION(:), &
         POINTER                                         :: res

      INTEGER                                            :: i, j, nderiv

      nderiv = 0
      DO i = 1, LEN(deriv_desc)
         IF (deriv_desc(i:i) == '(') THEN
            nderiv = nderiv + 1
         END IF
      END DO

      ALLOCATE (res(nderiv))

      nderiv = 0
      DO i = 1, LEN(deriv_desc)
         IF (deriv_desc(i:i) == '(') THEN
            nderiv = nderiv + 1
            DO j = i + 1, LEN(deriv_desc)
               IF (deriv_desc(j:j) == ')') EXIT
            END DO
!tc: should we do a check on the derivative descriptions?
            res(nderiv) = deriv_desc(i + 1:j - 1)
         END IF
      END DO

   END SUBROUTINE create_split_derivative_desc

END MODULE xc_derivative_desc
