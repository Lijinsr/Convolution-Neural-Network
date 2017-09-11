train_GD <- function(theta, datas, num_classes, filter_dim,
                      num_filters, pool_dim, preds,test,test_label, label,...){
  option <- list(...)
  if(is.null(option$epochs)) stop("please fill eopchs argument")
  if(is.null(option$alph)) stop("please fill all alpha argument")
  if(is.null(option$minibatch)) stop("please fill all minibatch argument")
  if(is.null(option$momentum)) option$momentum = 0
  k <- length(label)
  h <- test_label
  j <- sort(as.numeric(levels(as.factor(h))), index.return = T)
  for(i in 1:length(j$ix)){ 
    h[h == j$x[i]] <- j$ix[i]
  }
  initWc <- theta$Wc
  initWd <- theta$Wd
  #mom = 0
  #momIncrease = 36
  #Vwc <- array(0, dim = dim(theta$Wc))
  #Vwd <- array(0, dim = dim(theta$Wd))
  #Vbc <- array(0, dim = dim(theta$Bc))
  #Vbd <- array(0, dim = dim(theta$Bd))
  #Velocity <- list("Vwc"=Vwc,"Vwd"=Vwd,"Vbc"=Vbc,"Vbd"=Vbd)
  it = 0
  #plo <- matrix(0,nrow = option$epochs, ncol = 2)
  #plo <- data.frame(plo)
  ##plo <- (rep(0, times = option$epochs))
  train_acc <- (rep(0, times = (option$epochs * length(seq(from = 1, to = (k-option$minibatch+1), by = option$minibatch)))))
  test_acc <- (rep(0, times = (option$epochs * length(seq(from = 1, to = (k-option$minibatch+1), by = option$minibatch)))))
  plo <- matrix(0,nrow = (option$epochs * length(seq(from = 1, to = (k-option$minibatch+1), by = option$minibatch))), ncol = 2)
  plo <- data.frame(plo)
  for(e in 1:option$epochs){
    #rp <-1: k
    rp <- sample(k)
    for(s in seq(from = 1, to = (k-option$minibatch+1), by = option$minibatch)){
    it = it + 1
    #if(it == momIncrease) {
    # option$alph = 0.03#option$alph = option$alph/2 #mom = option$momentum
    #}
    #mb_data <- array(0,dim = dim(datas))# since minibatch is not under use
    mb_data<- datas[,,rp[s:(s+option$minibatch-1)]]
    mb_labels <-data.matrix(label[rp[s:(s+option$minibatch-1)]])
    grad <- cnn_cost(theta = theta,images = mb_data,label = mb_labels,
                     num_classes = num_classes, filter_dim = filter_dim,
                     num_filters = num_filters, pool_dim = pool_dim, preds = F)
    #grad <- cnn_cost(theta = theta,images = datas,label = label,
    #                 num_classes = num_classes, filter_dim = filter_dim,
    #                 num_filters = num_filters, pool_dim = pool_dim, preds = F)
    #Velocity$Vwc = mom * Velocity$Vwc + option$alph * grad$Wc_grad
    #Velocity$Vwd = mom * Velocity$Vwd + option$alph * grad$Wd_grad
    #Velocity$Vbc = mom * Velocity$Vbc + option$alph * grad$Bc_grad
    #Velocity$Vbd = mom * Velocity$Vbd + option$alph * grad$Bd_grad
    #theta$Wc = theta$Wc - Velocity$Vwc
    #theta$Wd = theta$Wd - Velocity$Vwd
    #theta$Bc = theta$Bc - Velocity$Vbc
    #theta$Bd = theta$Bd - Velocity$Vbd
    #num <- 1/dim(datas)[3]
    theta$Wc = theta$Wc - option$alph*grad$Wc_grad
    theta$Wd = theta$Wd - option$alph*grad$Wd_grad
    #theta$Bc = theta$Bc - option$alph*grad$Bc_grad
    #theta$Bd = theta$Bd - option$alph*grad$Bd_grad
    weight <- initWc - theta$Wc
    weight <- array(weight,c(dim(theta$Wc)[1]*dim(theta$Wc)[1],dim(theta$Wc)[3]))
    weight <- as.matrix(weight)*100
    jet.color <- colorRampPalette(c("yellow","red"))
    nbcol <- 100
    color <- jet.color(nbcol)
    weight_facet <- weight[-1,-1]+weight[-1,ncol(weight)]+
      weight[-nrow(weight),-1]+weight[-nrow(weight),-ncol(weight)]
    facetcol <- cut(weight_facet, nbcol)
    persp(x = 1:nrow(weight), y =1:ncol(weight), z= weight,
          main = "theta$Wc", theta = 45, phi = 45, expand = 1,
          box = F, col = color[facetcol], zlim = c(-max(weight), max(weight)))
    Sys.sleep(2)
    weight <- initWd - theta$Wd
    weight <- as.matrix(weight)*100
    weight_facet <- weight[-1,-1]+weight[-1,ncol(weight)]+
      weight[-nrow(weight),-1]+weight[-nrow(weight),-ncol(weight)]
    facetcol <- cut(weight_facet, nbcol)
    persp(x = 1:nrow(weight), y =1:ncol(weight), z= weight,
          main = "theta$Wc",theta = 45, phi = 45, col = color[facetcol],
          box = F, expand = 1, zlim = c(-max(weight), max(weight)))
    Sys.sleep(2)
    print(paste0("Epoch ",e," cost on iteration ",it," is ",grad$cost))
    plo[it,1] <- grad$cost
    plo[it,2] <- it
    pl <- ggplot(plo, aes(X2,X1))
    #par(mfrow = c(2,1))
    print(pl + geom_line())
    Sys.sleep(2)
    ##plo[e] <- grad$cost
    #####}
    #option$alph = option$alph/2
    opttheta <- theta
    sample_cost <- cnn_cost(theta = opttheta, images = mb_data, label = mb_labels,
                           num_classes = num_classes, filter_dim = filter_dim, 
                            num_filters = num_filters, pool_dim = pool_dim, preds = T)
    d <- mb_labels
    v <- sort(as.numeric(levels(as.factor(d))), index.return = T)
    for(i in 1:length(v$ix)){ 
      d[d == v$x[i]] <- v$ix[i]
    }
    train_acc[it] <- 100 * mean(sample_cost == d)
    sample_cost <- cnn_cost(theta = opttheta, images = test, label = test_label,
                           num_classes = num_classes, filter_dim = filter_dim, 
                            num_filters = num_filters, pool_dim = pool_dim, preds = T)
    test_acc[it] <- 100 * mean(sample_cost == h)
    #print(paste0("Accuraccy" , acc))
    #print(sample_cost)
    #plot(plo[1:e], type = "l", col = "black", ylim = c(0,1000))
    #lines(train_acc[1:e],col = "red")
    plot(train_acc[1:it],type = "l" ,col = "red", ylim = c(0,150))
    lines(test_acc[1:it],col="blue")
    Sys.sleep(2)
    }
  }
  #par(mfrow= c(1,1))
  opttheta <- theta
  return(opttheta)
  #return(grad)
}

