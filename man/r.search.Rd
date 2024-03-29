% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/r.search.R
\name{r.search}
\alias{r.search}
\title{Search rank magnitude of the best approximating matrix}
\usage{
r.search(
  x,
  r.min = 1,
  r.max = "auto",
  svd.method = c("tsvd", "rsvd"),
  rule.type = c("gic", "cv"),
  noise.var = 0,
  init = FALSE,
  init.mat = 0,
  maxit.rank = 1,
  nfolds = 5,
  thresh = 1e-05,
  maxit = 100,
  override = FALSE,
  control = list(...),
  ...
)
}
\arguments{
\item{x}{an \eqn{m} by \eqn{n} matrix with \code{NA}s.}

\item{r.min}{the start rank for searching. Default \code{r.min = 1}.}

\item{r.max}{the max rank for searching.}

\item{svd.method}{a character string indicating the truncated SVD method.
If \code{svd.method = "rsvd"}, a randomized SVD is used,
else if \code{svd.method = "tsvd"}, standard truncated SVD is used.
Any unambiguous substring can be given. Default \code{svd.method = "tsvd"}.}

\item{rule.type}{a character string indicating the information criterion rule.
If \code{rule.type = "gic"}, generalized information criterion rule is used,
else if \code{rule.type = "cv"}, cross validation is used.
Any unambiguous substring can be given. Default \code{rule.type = "gic"}.}

\item{noise.var}{the variance of noise.}

\item{init}{if init = FALSE(the default), the missing entries will initialize with mean.}

\item{init.mat}{the initialization matrix.}

\item{maxit.rank}{maximal number of iterations in searching rank. Default \code{maxit.rank = 1}.}

\item{nfolds}{number of folds in cross validation. Default \code{nfolds = 5}.}

\item{thresh}{convergence threshold, measured as the relative change in the Frobenius norm between two successive estimates.}

\item{maxit}{maximal number of iterations.}

\item{override}{logical value indicating whether the observed elements in \code{x} should be overwritten by its low-rank approximation.}

\item{control}{a list of parameters that control details of standard procedure, See \link{biscale.control}.}

\item{...}{arguments to be used to form the default control argument if it is not supplied directly.}
}
\value{
A list containing the following components
\item{\code{x.imp}}{the matrix after completion with the estimated rank.}
\item{\code{r.est}}{the rank estimation.}
\item{\code{rmse}}{the relative mean square error of matrix completion, i.e., training error.}
\item{\code{iter.count}}{the number of iterations.}
}
\description{
Estimate a preferable matrix rank magnitude for fitting a low-rank matrix approximation to a matrix with missing values.
The algorithm use GIC/CV to search the rank in a given range, and then fill the missing values with the estimated rank.
}
\examples{
################# Quick Start #################
m <- 100
n <- 100
r <- 10
x_na <- incomplete.generator(m, n, r)
head(x_na[, 1:6])
x_impute <- r.search(x_na, 1, 15, "rsvd", "gic")
x_impute[["r.est"]]
}
