!  MGH -
!
!  evaluate values and derivatives of one of the More'-Garbow-Hillstrom
!  test examples (Matlab interface to fortran MGH package by John L. Gardenghi,
!  see https://github.com/johngardenghi/mgh/blob/master/mgh.f08)

!  nick gould, july 16 2023

!  to initialize data structures for a particular problem
!   [ n, m, name, x_0, status ] = mgh( 'initial',  problem_number, dimension )
!
!  to evaluate the objective at x
!   [ f, status ] = mgh( 'eval_f', x )
!
!  to evaluate the gradient at x
!   [ g, status ] = mgh( 'eval_g', x )
!
!  to evaluate the Hessian at x
!   [ h, status ] = mgh( 'eval_h', x )
!
!  to evaluate the 3rd derivative tensor at x
!   [ t, status ] = mgh( 'eval_t', x )
!
!  to remove data structures after solution
!   [ status ] = mgh( 'final' )
!
!  Input -
!   problem_number: integer in the range [1,35] that specifies the example
!   x: evaluation point
!
!  Output -
!   n: problem dimension
!   m: residual dimension
!   name: name of problem indexed by problem_number
!   x_0: initial point
!   f: f(x), the value of the objective function at x
!   g: g(x), the value of the gradient of the objective function at x
!   h: h(x), the value of the Hessian of the objective function at x
!   t: t(x), the value of the 3-D tensor of the objective function at x
!   status: return code -
!           0. success
!          -1. problem_number out of range [1,35]
!          -2. input array not set
!          -3. evaluation error
!          -4. space allocation error
!          -5. initial call not made before other calls

#include <fintrf.h>

      subroutine mexFunction( nlhs, plhs, nrhs, prhs )
      use mgh ! use Gardenghi's mgh module
      use set_precision, only : rk ! define the precision used

! ------------------------- Do not change --------------------

!  Keep the above subroutine, argument, and function declarations for
!  use in all your fortran mex files.
!
      integer * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )
      logical :: mxIsChar, mxIsStruct
      mwSize :: mxGetString
      mwSize :: mxIsNumeric
      mwPointer :: mxCreateStructMatrix, mxGetPr

! -----------------------------------------------------------

      integer :: i, m, n, status, user_problem
      mwPointer :: arg2_in
      character ( len = 8 ) :: mode
      logical :: problem_set = .false.
      mwPointer :: i_pr, c_pr, vect_pr, status_pr
      real * 8, allocatable, save, dimension( : ) :: x8
      real ( kind = rk ) :: f
      real ( kind = rk ), allocatable, save, dimension( : ) :: x, g
      real ( kind = rk ), allocatable, save, dimension( : , : ) :: h
      real ( kind = rk ), allocatable, save, dimension( : , : , : ) :: t
      character( len = 60 ) :: name
      mwSize :: nn, mm, n2, n3
      mwSize :: dummy_mwSize__
      integer * 4 :: dummy_int4__
      integer, parameter :: mws_ = KIND( dummy_mwSize__ )
      integer, parameter :: int4_ = KIND( dummy_int4__ )
      integer * 4 :: status_4, user_problem_4, n_4, m_4
      character ( len = 50 ) :: txt
      real * 8 :: r8, r_vect( 1 )
      logical :: allocate_x

! interpret the input arguments, starting with the first key

     if ( nrhs < 1 ) &
       call mexErrMsgTxt( ' mgh requires at least 1 input argument' )

     if ( mxIsChar( prhs( 1 ) ) ) then
        i = mxGetString( prhs( 1 ), mode, 8 )
        if ( trim( mode ) /= 'final' ) then
          if ( nrhs < 2 ) &
           call mexErrMsgTxt( ' mgh requires at least 2 input arguments' )
          arg2_in = prhs( 2 )
!         call mexPrintf( txt )
        end if

!  see which call it is

        select case ( trim( mode ) )

!  initial call

        case ( 'initial' )
          n = - 1 ; m = - 1
          name = repeat( ' ', 60 )
          if ( mxIsNumeric( arg2_in ) == 0 ) &
            call mexErrMsgTxt( ' argument 2 must be an integer' )
          m = mxGetM( arg2_in ) ; n = mxGetN( arg2_in )
          if ( m /= 1 .or. n /= 1 ) &
            call mexErrMsgTxt( ' argument 2 must be an scalar' )
          call mxCopyPtrToReal8( mxGetPr( arg2_in ), r_vect, 1 )
          user_problem = int( r_vect( 1 ) )

          if ( user_problem > 0 .and. user_problem < 36 ) then
            problem_set = .true.

!  discover the problem dimensions and its name

            user_problem_4 = user_problem
            call mgh_set_problem( user_problem_4, status_4 )
            status = status_4
            if ( status == 0 ) then
              call mgh_get_dims( n_4, m_4 )
              n = n_4 ; m = m_4
              call mgh_get_name( name )

              if ( nrhs .ge. 3 ) then
                if ((mxIsNumeric( prhs( 3 ) ) == 0) .or. (mxGetM( prhs( 3 )) /= 1) .or. (mxGetN( prhs( 3 )) /= 1)) then
                  call mexErrMsgTxt( ' argument 3 must be an integer')
                end if
                call mxCopyPtrToReal8( mxGetPr( prhs( 3 ) ), r_vect, 1 )
                n = int( r_vect( 1 ) )
                n_4 = n
                call mgh_set_dims(n=n_4, flag=status_4)
                if ( status_4 /= 0 ) then
                  call mexErrMsgTxt( ' invalid dimension' )
                end if
              end if

!  allocate space and set the initial point

              if ( allocated( x ) ) deallocate( x, stat = status )
              if ( allocated( g ) ) deallocate( g, stat = status )
              if ( allocated( h ) ) deallocate( h, stat = status )
              if ( allocated( t ) ) deallocate( t, stat = status )
              allocate( x( n ), g( n ), h( n, n ), t( n, n, n ), &
                        stat = status )
!!            avoid bug in mgh_get_x0 by setting optional factor = 1.0_rk
              if ( status == 0 ) call mgh_get_x0( x, factor = 1.0_rk )
            end if

!  copy output data -

!  n
            plhs( 1 ) = mxCreateDoubleMatrix( 1, 1, 0 )
            r_vect( 1 ) = real( n, kind = kind( r8 ) )
            call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 1 ) ), 1_mws_ )

!  m
            plhs( 2 ) = mxCreateDoubleMatrix( 1, 1, 0 )
            r_vect( 1 ) = real( m, kind = kind( r8 ) )
            call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 2 ) ), 1_mws_ )

!  name
            plhs( 3 ) = mxCreateCharMatrixFromStrings( 1_mws_, trim( name ) )

!  x
            nn = n
            plhs( 4 ) = mxCreateDoubleMatrix( nn, 1, 0 )
            call mxCopyReal8ToPtr( x, mxGetPr( plhs( 4 ) ), nn )

!  status
            plhs( 5 ) = mxCreateDoubleMatrix( 1, 1, 0 )
            r_vect( 1 ) = real( status, kind = kind( r8 ) )
            call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 5 ) ), 1_mws_ )

          else
            call mexErrMsgTxt( ' problem out of range' )
            status = - 1
          end if

!  evaluate f, g, h or t call

        case ( 'eval_f', 'eval_g', 'eval_h', 'eval_t' )

          if ( problem_set ) then
            if ( mxIsNumeric( arg2_in ) == 0 ) &
               call mexErrMsgTxt( ' argument 2 should be the vector x' )

!  recall the problem dimensions

            call mgh_get_dims( n_4, m_4 )
            n = n_4 ; m = m_4

!  make space for x, and extract it from the input pointer

            mx = mxGetM( arg2_in ) ; nx = mxGetN( arg2_in )
            if ( mx /= n .or. nx /= 1 ) &
              call mexErrMsgTxt( ' argument 2 must be have dimensions n x 1' )
            if ( allocated( x8 ) ) then
              if ( size( x8 ) >= n ) then
                allocate_x = .false.
              else
                allocate_x = .true.
                deallocate( x8, stat = status )
              end if
            else
              allocate_x = .true.
            end if
            if ( allocate_x ) allocate( x8( n ), stat = status )
            call mxCopyPtrToReal8( mxGetPr( arg2_in ), x8, n )
            x = x8

!  which of the f, g, h or t calls is it?

            select case ( trim( mode ) )

!  evaluate f call

            case ( 'eval_f' )
              call mgh_evalf( x, f, status_4 )
              status = status_4

!  copy f to output

              plhs( 1 ) = mxCreateDoubleMatrix( 1, 1, 0 )
              r_vect( 1 ) = f
              call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 1 ) ), 1_mws_ )

!  evaluate g call

            case ( 'eval_g' )
!              if ( .not. allocated( g ) ) &
!                allocate( g( n ), stat = status )
              call mgh_evalg( x( : n ), g( : n ), status_4 )
              status = status_4

!  copy g to output

              plhs( 1 ) = mxCreateDoubleMatrix( n, 1, 0 )
              call mxCopyReal8ToPtr( g, mxGetPr( plhs( 1 ) ), n )

!  evaluate h call

            case ( 'eval_h' )
!              if ( .not. allocated( h ) ) &
!                allocate( h( n, n ), stat = status )
              call mgh_evalh( x, h, status_4 )
              status = status_4

!  symmetrize h as mgh only calculates the upper triangle

              do j = 1, n
                do i = 1, j - 1
                  h( j, i ) = h( i, j )
                end do
              end do

!  copy h to output

              nn = n ; n2 = nn * nn
              plhs( 1 ) = mxCreateNumericArray( 2_mws_, (/ nn, nn /), &
                                 mxClassIDFromClassName( 'double' ), 0_int4_ )
              call mxCopyReal8ToPtr( [ h ], mxGetPr( plhs( 1 ) ), n2 )

!  evaluate t call

            case ( 'eval_t' )
!              if ( .not. allocated( t ) ) &
!                allocate( t( n, n, n ), stat = status )
              call mgh_evalt( x, t, status_4 )
              status = status_4

!  symmetrize t as mgh only calculates the upper "triangle"

              do k = 1, n
                do j = 1, k
                  do i = 1, j
!                    t( i, j, k ) = t( i, j, k )
                    t( i, k, j ) = t( i, j, k )
                    t( j, i, k ) = t( i, j, k )
                    t( j, k, i ) = t( i, j, k )
                    t( k, i, j ) = t( i, j, k )
                    t( k, j, i ) = t( i, j, k )
                  end do
                end do
              end do

!  copy t to output

              nn = n ; n3 = nn * nn * nn
              plhs( 1 ) = mxCreateNumericArray( 3_mws_, (/ nn, nn, nn /), &
                                 mxClassIDFromClassName( 'double' ), 0_int4_ )
              call mxCopyReal8ToPtr( [ t ], mxGetPr( plhs( 1 ) ), n3 )
            end select

!  copy status to output

              plhs( 2 ) = mxCreateDoubleMatrix( 1, 1, 0 )
              r_vect( 1 ) = real( status, kind = kind( r8 ) )
              call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 2 ) ), 1_mws_ )
          else
            status = - 5
          end if


!  final call

        case ( 'final' )
          if ( .not. problem_set ) then
            status = - 1
          else
            problem_set = .false.
            deallocate( x8, x, g, h, t, stat = status )
          end if

!  copy status to output

          plhs( 1 ) = mxCreateDoubleMatrix( 1, 1, 0 )
          r_vect( 1 ) = real( status, kind = kind( r8 ) )
          call mxCopyReal8ToPtr( r_vect, mxGetPr( plhs( 1 ) ), 1_mws_ )

!  unknown call

        case default
          call mexErrMsgTxt( ' argument 1 key not recognised' )
        end select
     else
       call mexErrMsgTxt( ' argument 1 not a string' )
     end if


     end subroutine mexFunction
