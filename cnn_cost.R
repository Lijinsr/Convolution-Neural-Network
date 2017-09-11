cnn_cost <- function(theta, images, label, num_classes, filter_dim,
                     num_filters, pool_dim,preds =F) {
  #options(digits = 2)
  image_dim <- dim(images)[1]
  num_images <- dim(images)[3]
  Wc <- theta$Wc
  Wd <- theta$Wd
  Bc <- theta$Bc
  Bd <- theta$Bd
  Wc_grad <- array(0, dim = dim(Wc))
  Wd_grad <- array(0, dim = dim(Wd))
  Bc_grad <- array(0, dim = dim(Bc))
  Bd_grad <- array(0, dim = dim(Bd))
  activations <- cnn_convolve(filter_dim = filter_dim, num_filters = 
                                num_filters, images = images, w = Wc,
                              b = Bc)
  activations_pooled <- cnn_pool(pool_dim = pool_dim,
                                 convolved_features = activations)
  conv_dim <- image_dim- filter_dim+1
  out_dim <- conv_dim/pool_dim
  hidden_size <- out_dim^2 * num_filters
  activations_pooled <- aperm(array(activations_pooled,c(num_images,
                                                         hidden_size)))
  Bd <- array(rep(Bd, times= num_images), c(num_classes,num_images))
  op_term <- tanh((Wd %*% activations_pooled) + Bd)
  repmat <- as.factor(label)
  label_mask_mat <- matrix(levels(repmat), nrow = 1)
  label_check <- t(label)
  M1 <- (label_check == label_mask_mat[1]) * 1 
  for(i in 2:ncol(label_mask_mat)){
    M2 <- (label_check == label_mask_mat[i]) * 1
    M1 <- rbind(M1,M2)
  }
  required <- M1 + (((M1 == 0)*1)* -1)
  error <- (required - op_term)
  cost <- sum(error^2)
  if (preds == T){
    preds <- op_term
    preds <- apply(preds,2,which.max)
    preds <- t(t(preds))
    #grad <- 0
    #return(preds)
    #return(op_term)
    return(required)
  }
  fpo <- -2 * error * (1 - op_term^2)
  fp <-t(Wd) %*% fpo
  for(image_num in 1:num_images) {
    delta_source <- aperm(array(fp[, num_images], c(num_filters,out_dim,out_dim)))
    im <- images[,,image_num]
    for(filter_num in 1: num_filters) {
      delta_pool <- 1 /(pool_dim^2) *  kronecker(delta_source[,,filter_num],
                                                 array(1,c(pool_dim,pool_dim)))
      delta_pool <- delta_pool* (1 - (activations[,,filter_num,image_num])^2)
      L <- array(0, c(filter_dim,filter_dim))
      for(i in 1:filter_dim) {
        for(j in 1:filter_dim) {
          L[i,j] <- convolve(im[i:(i+conv_dim-1), j:(j+conv_dim-1)],
                                  delta_pool, type = "filter")
        }
      }
      Wc_grad[,,filter_num] <- Wc_grad[,,filter_num]+L
      Bc_grad[filter_num] <- Bc_grad[filter_num]+sum(delta_pool)
    }
  }
  Wd_grad <- (fpo %*% t(activations_pooled))
  Bd_grad <- t(t(rowSums(fpo)))
  grad <- list("Wc_grad"= Wc_grad, "Wd_grad"= Wd_grad, 
               "Bc_grad" = Bc_grad, "Bd_grad"=Bd_grad, "cost"= cost)
  return(grad)
}

