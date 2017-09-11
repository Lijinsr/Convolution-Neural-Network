cnn_convolve <- function(filter_dim, num_filters, images, w,b) {
  #options(digits = 2)
  num_images <- dim(images)[3]
  image_dim <- dim(images)[1]
  conv_dim <- image_dim - filter_dim+1
  col_dim <- dim(images)[2] - filter_dim +1
  convolved_features <- array(0, c(conv_dim, conv_dim, num_filters,
                                   num_images))
  for(image_num in 1:num_images) {
    for(filter_num in 1:num_filters) {
      convolved_image <- array(0, c(conv_dim, conv_dim))
      filters <- w
      x <- images[,,image_num]
      y <- filters[,,filter_num]
      for(i in 1:conv_dim) {
        for(j in 1: col_dim) {
          convolved_image[i,j] <- convolve(x[i:(i+filter_dim-1),
                                             j:(j+filter_dim-1)], y, 
                                           type = "filter")
        }
      }
      convolved_image <- convolved_image + b[filter_num]
      convolved_image <- tanh(convolved_image)
      
      convolved_features[,,filter_num, image_num] <- convolved_image
      #return(convolved_features)
    }
  }
}

