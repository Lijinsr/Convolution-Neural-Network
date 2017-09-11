cnn_Initparam <- function(Image_dim, filter_dim, num_filters,
                          pool_dim, num_classes){
  stopifnot(filter_dim < Image_dim)
  #options(digits = 2)
  #Wc <- array((rnorm(10)), c(filter_dim, filter_dim, num_filters))
  Wc <- array((runif(10)), c(filter_dim, filter_dim, num_filters))
  out_dim <- Image_dim - filter_dim + 1
  stopifnot(out_dim %% pool_dim == 0)
  out_dim <- out_dim/pool_dim
  hidden_size <- out_dim^2 * num_filters
  #r <- sqrt(6) / sqrt(num_classes+hidden_size+1)
  Wd <- array(rnorm(10), c(num_classes, hidden_size))
  Wd <- array(runif(10), c(num_classes, hidden_size))
  Bc <- array(0.9, c(num_filters,1))
  Bd <- array(0.9,c(num_classes,1))
  theta <- list("Wc" = Wc, "Wd" = Wd, "Bc" = Bc, "Bd" = Bd)
  return(theta)
}

