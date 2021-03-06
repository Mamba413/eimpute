#' @title Data standardization
#'
#' @description Standardize a matrix rows and/or columns to have zero mean or unit variance
#'
#' @param x an \eqn{m} by \eqn{n} matrix possibly with \code{NA}s.
#' @param thresh.sd convergence threshold, measured as the relative change in the Frobenius norm between two successive estimates.
#' @param maxit.sd maximum number of iterations.
#' @param control a list of parameters that control details of standard procedure. See \link{stad.control}.
#' @param ... arguments to be used to form the default control argument if it is not supplied directly.
#'
#' @return A list is returned
#' \item{\code{x.st}}{The matrix after standardization.}
#' \item{\code{alpha}}{The row mean after iterative process.}
#' \item{\code{beta}}{The column mean after iterative process.}
#' \item{\code{tau}}{The row standard deviation after iterative process.}
#' \item{\code{gamma}}{The column standard deviation after iterative process.}
#'
#' @references Hastie, Trevor, Rahul Mazumder, Jason D. Lee, and Reza Zadeh. Matrix completion and low-rank SVD via fast alternating least squares. The Journal of Machine Learning Research 16, no. 1 (2015): 3367-3402.
#'
#' @export
#' @examples
#' ################# Quick Start #################
#' m <- 100
#' n <- 100
#' r <- 10
#' x_na <- incomplete.generator(m, n, r)
#'
#' ###### Standardize both mean and variance
#' xs <- biscale(x_na)
#'
#' ###### Only standardize mean ######
#' xs_mean <- biscale(x_na, row.mean = TRUE, col.mean = TRUE)
#'
#' ###### Only standardize variance ######
#' xs_std <- biscale(x_na, row.std = TRUE, col.std = TRUE)
biscale <- function(x, thresh.sd = 1e-05, maxit.sd = 100, control = list(...), ...){
  m <- nrow(x)
  n <- ncol(x)
  extraArgs <- control
  if (length(extraArgs)) {
    controlargs <- names(formals(stad.control)) # legal arg names
    indx <- match(names(extraArgs), controlargs, nomatch = 0L)
    if (any(indx == 0L))
      stop(gettextf("Argument %s not matched",
                    names(extraArgs)[indx == 0L]),
           domain = NA)
  }
  controls <- do.call("stad.control", control)
  row.mean <- controls$row.mean
  row.std <- controls$row.std
  col.mean <- controls$col.mean
  col.std <- controls$col.std
  if(row.mean + row.std + col.mean + col.std > 0){
    xna <- is.na(x)

    m_ob <- n - rowSums(xna)
    n_ob <- m - colSums(xna)
    m_ob[which(m_ob == 0)] <- 1e-08
    n_ob[which(n_ob == 0)] <- 1e-08

    x_ind <- which(!is.na(x))
    x_alt <- x
    x_alt[-x_ind] <- 0

    # alpha <- rowMeans(x_alt)
    # beta <- colMeans(x_alt)
    # tau <- apply(x_alt, 1, sd)
    # gamma <- apply(x_alt, 2, sd)
    if(is.numeric(row.mean)){
      if(length(row.mean) == m){
        alpha <- row.mean
        row.mean <- FALSE
      }else{
        stop("length of 'row.mean' must equal the number of rows of 'x'")
      }
    }else{
      alpha <- rep(0, m)
    }

    if(is.numeric(col.mean)){
      if(length(col.mean) == n){
        beta <- col.mean
        col.mean <- FALSE
      }else{
        stop("length of 'col.mean' must equal the number of columns of 'x'")
      }
    }else{
      beta <- rep(0, n)
    }

    if(is.numeric(row.std)){
      if(length(row.std) == m){
        tau <- row.std
        row.std <- FALSE
      }else{
        stop("length of 'row.std' must equal the number of rows of 'x'")
      }
    }else{
      tau <- rep(1, m)
    }

    if(is.numeric(col.std)){
      if(length(col.std) == n){
        gamma <- col.std
        col.std <- FALSE
      }else{
        stop("length of 'col.std' must equal the number of columns of 'x'")
      }
    }else{
      gamma <- rep(1, n)
    }

    temp <- matrix(rep(0, n*m), nrow = m, ncol = n)
    iter <- 0
    err <- 100
    while((err > thresh.sd) & (iter < maxit.sd)){
      alpha_old <- alpha
      beta_old <- beta
      tau_old <- tau
      gamma_old <- gamma
      tau[which(tau == 0)] <- 1e-08
      gamma[which(gamma == 0)] <- 1e-08
      alpha_m <- matrix(rep(alpha, n), nrow = m)
      beta_m <- t(matrix(rep(beta, m), nrow = n))

      if(col.mean){
        temp[x_ind] <- x_alt[x_ind] - alpha_m[x_ind]
        beta <- colSums(temp/tau)/sum(1/tau) # col mean
      }

      if(row.mean){
        temp[x_ind] <- x_alt[x_ind] - beta_m[x_ind]
        alpha <- colSums(t(temp)/gamma)/sum(1/gamma) # row mean
      }

      if(col.std | row.std){
        temp[x_ind] <- (x_alt[x_ind] - alpha_m[x_ind] - beta_m[x_ind])^2
        if(col.std){
          gamma <- sqrt(colSums(temp/tau^2)/n_ob) # col var
          gamma[which(gamma == 0)] <- 1e-08
        }

        if(row.std){
          tau <- sqrt(colSums(t(temp)/gamma^2)/m_ob) # row var
          tau[which(tau == 0)] <- 1e-08
        }
      }

      err <- max(sum(alpha - alpha_old)^2, sum(beta - beta_old)^2, sum(tau - tau_old)^2, sum(gamma - gamma_old)^2)
      iter <- iter + 1
    }
    alpha_m <- matrix(rep(alpha, n), nrow = m)
    beta_m <- t(matrix(rep(beta, m), nrow = n))
    x_st <- (x_alt - alpha_m - beta_m)/(tau %*% t(gamma))
    x_st[-x_ind] <- NA
  }else{
    x_st <- x
    alpha <- rep(0, m)
    beta <- rep(0, n)
    tau <- rep(1, m)
    gamma <- rep(1, n)
  }

  list(x.st = x_st, alpha = alpha, beta = beta, tau = tau, gamma = gamma)
}


#' @title Control for standard procedure
#'
#' @description Various parameters that control aspects of the standard procedure.
#'
#'
#' @param row.mean if \code{row.mean = TRUE} (the default), row centering will be performed resulting in a matrix with row means zero.
#' If \code{row.mean} is a vector, it will be used in the iterative process.
#' If \code{row.mean = FALSE} nothing is done.
#' @param row.std if \code{row.std = TRUE} , row scaling will be performed resulting in a matrix with row variance one.
#' If \code{row.std} is a vector, it will be used in the iterative process.
#' If \code{row.std = FALSE} (the default) nothing is done.
#' @param col.mean similar to \code{row.mean}.
#' @param col.std similar to \code{row.std}.
#'
#' @export
stad.control <- function(row.mean = FALSE, row.std = FALSE, col.mean = FALSE, col.std = FALSE) {
  list(row.mean = row.mean, row.std = row.std, col.mean = col.mean, col.std = col.std)
}

