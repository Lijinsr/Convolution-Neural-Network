cnn_pool <- function(pool_dim, convolved_features){
  #options(digits = 2)
  num_images <- dim(convolved_features)[4]
  num_filters <- dim(convolved_features)[3]
  convolved_dim <- dim(convolved_features)[1]
  col_dim <- dim(convolved_features)[2]
  
  pooled_features <- array (0, c((convolved_dim/pool_dim),
                                 (convolved_dim/pool_dim), num_filters,
                                 num_images))
  for(image_num in 1:num_images){
    for(filter_num in 1:num_filters){
      pooled_image <- array(0,c((convolved_dim/pool_dim),
                                (convolved_dim/pool_dim)))
      x <- convolved_features[,,filter_num,image_num]
      k <- 0; l <- 0; m <- 0
      for(i in seq(from = 1, to = convolved_dim, by = pool_dim)) {
        k=k+1; l=1
        for(j in seq(from=1, to=col_dim, by=pool_dim)) {
          m <- mean(x[i:(i+pool_dim-1), j:(j+pool_dim-1)])
          while(k<=(convolved_dim/pool_dim)) {
            while(l<=(convolved_dim/pool_dim)) {
              pooled_image[k,l] <- m
              break
            }
            l=l+1
            break
          }
        }
      }
      pooled_features[,,filter_num,image_num] <- pooled_image
      #return(pooled_features)
    }
  }
}

