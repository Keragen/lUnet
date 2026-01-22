
###########################################################
###Run Lunet to preprocess the files from the microscope###
###Then run spoton to quantify the signals              ###
###########################################################


#masks are generated in 1024 for better performance then the size is matched to the original picture in spoton

library(keras)
library(magick)
library(shotGroups)
library(EBImage)
library(scales)
library(dplyr)
library(gtools)
library(stringr)
library(foreach)
library(doParallel) #for mac and linux



unetpathcyto<-"/shared/space2/projects/genearray/tools/VALID-Attila-unet-CYTO_001.h5"
unetpathnuc <-"/shared/space2/projects/genearray/tools/VALID-Attila-unet-NUC_001.h5"

###########to setup#################################
#folder to save results into (OUTPUT)
targetmovie<-"/shared/space2/projects/genearray/tools/moviesUnet/AO2022-35/"

#folder to analyze (INPUT)
movieroot<-"/shared/space2/projects/genearray/Attila/AO2022-35/"



cl <- makeCluster(5)  #set number of threads
####################################################



registerDoParallel(cl)


input_sizeX <- 1024 #2048 #1024  #1024fix
input_sizeY <- 1024 #2048 #1024  #1024fix


# Loss function -----------------------------------------------------

K <- backend()

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) /
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
}

bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}

# # U-net 1024 -----------------------------------------------------


get_unet_1024 <- function() {    

  input_shape = c(input_sizeX,input_sizeY,3)
  num_classes = 1 #3 for color masks
  
  inputs <- layer_input(shape = input_shape)
  # 1024


  down0b <-inputs %>%
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down0b_pool <- down0b %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8

  down0a <-down0b_pool %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down0a_pool <- down0a %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  #16

  down0 <-down0a_pool %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down0_pool <- down0 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  #32

  down1 <- down0_pool %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 64

  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 32

  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16

  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8

  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # center

  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 16

  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 32

  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 64

  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 128

  up0 <- up1 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down0, .), axis = 3)} %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 256

  up0a <- up0 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down0a, .), axis = 3)} %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 512

  up0b <- up0a %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down0b, .), axis = 3)} %>%
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 1024

  classify <- layer_conv_2d(up0b,
                            filters = num_classes,
                            kernel_size = c(1, 1),
                            activation = "sigmoid")


  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )

  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.0001),
    loss = bce_dice_loss,
    metrics = custom_metric("dice_coef", dice_coef)
  )

  return(model)
}


# ################################################
# ##############pred for 1 image##################
# ################################################

modelnuc <-get_unet_1024()
modelcyto<-get_unet_1024()

load_model_weights_hdf5(modelnuc ,unetpathnuc)
load_model_weights_hdf5(modelcyto,unetpathcyto)



fileext<-"jpg"    #ouput format
newposition<-data.frame(obj=integer(),
                        x=double(),
                        y=double(),
                        stringsAsFactors=FALSE)


biometry<<-data.frame(
  file=character(),
  name=character(),
  surface=double(),
  intensity=double(),
  corrintensity=double(),
  length=double(),
  width=double(),
  elongation=double(),
  orientation=double(),
  infolding=double(),
  stringsAsFactors=FALSE)




testimageRead <- function(image_file,
                          target_width = input_sizeX,
                          target_height = input_sizeY) {
  img <- image_read(image_file)
  img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
  return(img)
}

testimg2arr <- function(image,
                        target_width = input_sizeX,
                        target_height = input_sizeY) {
  result <- aperm(as.numeric(image[[1]])[, , 1:3], c(2, 1, 3)) # transpose
  array_reshape(result, c(1, target_width, target_height, 3))

}



erode<-function(img){
  #use this to erode mask (and break small links between objects)
  kern = makeBrush(9, shape="disc")    #box, disc, diamond, gaussian, line
  img_erode= erode(as_EBImage(img), kern)
  display(img_erode)
}


getobjcenter<-function(nmask,mode){
  
  if(file.exists("positions0.txt")){
    oldposition<-read.table("positions0.txt",stringsAsFactors = FALSE,check.names = FALSE,header = FALSE)
    names(oldposition)<-c("obj","x","y")
  }
  
  #####################compute center of objects for tracking##################################
  allobj<-table(nmask@.Data)
  newposition<-data.frame(obj=integer(),
                          x=double(),
                          y=double(),
                          stringsAsFactors=FALSE)
  for(obji in as.integer(names(allobj))){
    singleobj <- nmask@.Data== obji    #coordinates for obj obji
    singleobjcoords<-which(singleobj==TRUE, arr.ind=T) #get center of object to write stuffs on the picture
    
    if(obji>0){  #skip background
      
      pca<-prcomp(singleobjcoords)
      center<-pca$center
      center.x<-center[1]
      center.y<-center[2]
      
      # print (paste0(obji," ",center.x," ",center.y))
      
      # save position for tracking
      newposition<-rbind(newposition,c(obji,center.x,center.y))
      
    }
  }
  
  if(file.exists("positions0.txt")){
    names(newposition)<-c("obj","x","y")
    newposition<-rbind(newposition,oldposition)
    newposition<-newposition[!duplicated(newposition[,"obj"]),]
  }
  
  write.table(newposition,"positions.txt", row.names = FALSE, col.names = FALSE, append = FALSE)
  
  if(mode==1) #update this file only with corrected tracking
    write.table(newposition,"positions0.txt", row.names = FALSE, col.names = FALSE, append = FALSE)
  
  #############################################################################################
}

water<-function(img,watersize,objsize){
  
  
  img<-normalize(img)
  nmask = watershed( distmap(img), watersize) #0.01,1
  
  
  ###if an object is too small it will be erased from the mask####
  minsurface<-objsize
  allobj<-table(nmask@.Data)  #each number correspond to a distinct object
  toremove<-allobj[allobj<minsurface]
  nmask = rmObjects(nmask, names(toremove), reenumerate=TRUE)
  ############
  
  #to remove small gaps inside masks
  nmask<-fillHull(nmask)
  
  #assign a color to objects
  # nmask<-colorLabels(nmask)
  
  #save obj center for tracking
  getobjcenter(nmask,0)
  
  
  return(nmask)
}

polygonsurface<-function(points){
  points<-rbind(points,points[1,])
  area<-0
  for(p in seq(1:(nrow(points)-1)) ){
    
    pt1<-points[p,]
    pt2<-points[p+1,]
    
    segt<-pt1[1]*pt2[2]-pt1[2]*pt2[1]
    area<-area+segt
  }
  area<-abs(area/2)
  return(area)
}



shapeannot<-function(nmask,imgbio, testimage_file, style){
  allobj<-table(nmask@.Data)  #each number correspond to a distinct object
  
  nmask<-nmask[ ,ncol(nmask):1 ]  #reverse mask to match picture coord system
  
  
  imgbio<-imgbio[ ,ncol(imgbio):1, ]
  imgbio<-channel(imgbio,"gray")
  
  

  
  for(obji in as.integer(names(allobj))){
    
    val<-allobj[names(allobj)==obji]
    
    singleobj <- nmask@.Data== obji    #coordinates for obj obji
    singleobjcoords<-which(singleobj==TRUE, arr.ind=T) #get center of object to write stuf on the picture
    

    if(nrow(singleobjcoords)<=1)
      next()
    
    
    if(obji>0){  #skip background
      #PCA
      
      pca<-prcomp(singleobjcoords)

      r<-pca$rotation
      

      #PC1
      slope<-r[2,1]/r[1,1]
      



      #PC2

      center<-pca$center
      center.x<-center[1]
      center.y<-center[2]
      
      
      H  <- chull(singleobjcoords)
      if(style==1)
        polygon(singleobjcoords[H, ], col="red", density = 0)
      chullsurf<-polygonsurface(singleobjcoords[H, ])
      infolding<- 100-(val/chullsurf)*100


      #compute bounding box coords from PCA
      minx<-min(pca$x[,1])
      maxx<-max(pca$x[,1])
      miny<-min(pca$x[,2])
      maxy<-max(pca$x[,2])
      bboxpoints<-matrix(nrow = 4,ncol=2,data = c(minx,miny, minx,maxy, maxx,maxy, maxx,miny),byrow = TRUE)
      mu = colMeans(singleobjcoords)
      nComp = 2
      Xhat = bboxpoints[,1:nComp] %*% t(pca$rotation[,1:nComp])
      Xhat = scale(Xhat, center = -mu, scale = FALSE)
      bboxcoords<-Xhat[,1:2]
      if(style==1){
        #show bounding box
        drawBox2(bboxcoords, fg='yellow', pch=4, cex=2)
        
        #print PC1 et 2 (orientations)
        segments(mean(bboxcoords[1:2,1]), mean(bboxcoords[1:2,2]), mean(bboxcoords[3:4,1]), mean(bboxcoords[3:4,2]), col="yellow")
        segments(mean(bboxcoords[2:3,1]), mean(bboxcoords[2:3,2]), mean(bboxcoords[c(4,1),1]), mean(bboxcoords[c(4,1),2]), col="red")
      }
      

      alldist <- as.matrix(dist(bboxcoords, method = "euclidean"))
      length<-max(alldist[2,1],alldist[4,1])
      width<-min(alldist[2,1],alldist[4,1])
      elongation<-width/length
      
      
    }
    

    singleobj <- imgbio[singleobj] #get pixel values from the real bio image
    meanint<-round(mean(singleobj)*100,2)

    
    if(obji==0){
      bgint<-meanint
    }else{

      
      if(style==1){
        labelobj<-paste0("object ",obji,"\n","surface: ",val," px\n" ) #text to print on picture
        text(center.x, center.y, labelobj,col = "white", cex=1)  #0.5
      }


      if(style==1)
        obji<-paste0("nucleus",obji)
      else
        obji<-paste0("cytoplasm",obji)
      
      dataobj<-list(testimage_file, as.character(obji),val,meanint,length,width,elongation,round(slope,2),infolding)
      dataobj<-as.data.frame(dataobj)
      names(dataobj)<-c("file","name","surface","intensity","length","width","elongation","orientation","infolding")
      biometry<<-rbind(biometry,dataobj)
      names(biometry)<<-c("file","name","surface","intensity","length","width","elongation","orientation","infolding")

    }
    
  }
}


createmask<-function(imagefilepath, testimage_file,Voronoilambda,watershedsize1,watershedsize2,watersizegap1,watersizegap2,nucmaskthreshold,cytomaskthreshold,nucblend,cytoblend,nuccolorlabel,cytocolorlabel,extractnucfromcyto,dobackground){
  

  test<-testimageRead(imagefilepath)
  
  test2<-testimg2arr(test)
  
  preds2nuc <-modelnuc %>% predict(test2)  #for dual unet
  preds2cyto<-modelcyto %>% predict(test2)  #for dual unet
  
  
  
  predtmpnuc <-  array_reshape(preds2nuc,  c(input_sizeX,input_sizeY,1))
  predtmpcyto<-  array_reshape(preds2cyto, c(input_sizeX,input_sizeY,1))
  
  
  
  ################
  #removes smaller nuclei (false positives)
  predtmpnuc<- ifelse(predtmpnuc > nucmaskthreshold , 1, 0) 
  nmask<-watershed( distmap(predtmpnuc), 2 )
  allobj<-table(nmask@.Data)  #each number correspond to a distinct object
  allobj<-allobj[names(allobj)!="0"]
  minsurface<-mean(allobj)/5      #if 5x smaller than average then probably fake
  toremove<-allobj[allobj<minsurface]
  predtmpnuc = rmObjects(nmask, names(toremove), reenumerate=F)
  #####################
  
  
  
  predtmp<-array(dim=c(input_sizeX,input_sizeY,3))
  predtmp[,,3]<-0 #nothing is supposed to be in the blue channel
  predtmp[,,2]<-predtmpnuc
  predtmp[,,1]<-predtmpcyto

  
  predtmp[,,2]<-ifelse(predtmp[,,2] > nucmaskthreshold , 1, 0)    
  
  predtmp[,,1]<-ifelse(predtmp[,,1] > cytomaskthreshold & predtmp[,,2]<1 , 1, 0)       #for dual lunet only. removes nuc from cyto
  
  
  

  
  
  
  
  if(dobackground==TRUE){
    #watersizegap1<-2
    #watersizegap2<-200
    #compute gaps
    gaps<-as.matrix(predtmp[,,1])+as.matrix(predtmp[,,2])
    gaps<-ifelse(gaps > 0, 0, 1)
    gaps <- watershed(distmap(gaps),watersizegap1)  #,2,2?
    gaps <- fillHull(gaps)
    ###erase small gaps####
    minsurface<-watersizegap2
    allobj<-table(gaps@.Data)  #each number correspond to a distinct object
    toremove<-allobj[allobj<minsurface]
    gaps = rmObjects(gaps, names(toremove), reenumerate=FALSE)
    ############
    predtmp[,,1][gaps==0]<-1  # erase small gaps from mask

  }
  
  
  
  
  
  
  #removes overlaps.No red allowed where green is detected
  if(extractnucfromcyto==TRUE){
    predtmp[,,1][predtmp[,,2]>0]<-0
  }

  
  
  img<-magick::image_read(predtmp) %>% magick::image_scale(paste0(input_sizeX,"x",input_sizeY,"!"))
  
  
  img<-image_rotate(img,90) #for color mask
  img<-image_flop(img)      #for color mask
  
  
  #save new mask
  image_write(img,paste(resultmaskspath,testimage_file,sep=""),fileext)
  
  
  
  
  
  
  
  #blend picture with mask     
  imgred<-predtmp
  imgred[,,2]<-0
  imgred[,,3]<-0
  imgred<-image_read(imgred)
  imgred<-image_rotate(imgred,90)
  imgred<-image_flop(imgred)
  
  imggreen<-predtmp
  imggreen[,,1]<-0
  imggreen[,,3]<-0
  imggreen<-image_read(imggreen)
  imggreen<-image_rotate(imggreen,90)
  imggreen<-image_flop(imggreen)
  
  imgred  <-image_transparent(imgred, "#000000", fuzz = 10)
  imggreen<-image_transparent(imggreen, "#000000", fuzz = 10)
  
  

  composite<-image_composite(test,imggreen,operator = "blend",compose_args = nucblend)
  composite<-image_composite(composite,imgred,operator = "blend",compose_args = cytoblend)
  
  
  if(file.exists("positions.txt") ){
    lastposition<-read.table("positions.txt",stringsAsFactors = FALSE,check.names = FALSE,header = FALSE)
  }else{
    lastposition<-data.frame()
  }
  
  switch<-0
  if(nrow(lastposition)>0){
    switch<-1
  }
  
  watermask<-water(predtmp[,,2], watershedsize1, watershedsize2)

  
  
  # ##########################################remap of object names for tracking################################################
  if(switch==1){
    

    newposition<-read.table("positions.txt",stringsAsFactors = FALSE,check.names = FALSE,header = FALSE)
    
    names(lastposition)<-c("obj","X","Y")
    names(newposition)<-c("obj","X","Y")
    
    
    remap<-data.frame(old=integer(),new=integer(),dist=double())
    for(pos in 1:nrow(newposition)){
      for(pos2 in 1:nrow(lastposition)){
        x1<- newposition[pos,"X"]
        y1<- newposition[pos,"Y"]
        x2<-lastposition[pos2,"X"]
        y2<-lastposition[pos2,"Y"]
        dist<-sqrt((x1-x2)^2 + (y1-y2)^2)
        remap<-rbind(remap,c(lastposition[pos2,"obj"], newposition[pos,"obj"], dist))
      }
    }
    names(remap)<-c("last","new","dist")

    
    remap<-remap %>%
      group_by(new) %>%
      slice(which.min(dist))
    
    remap<-remap[order(remap$dist),] #sort by increasing distance
    

    thr     <- 50 #meandist+(5*sddist)

    
    dupli<-remap[duplicated(remap$last),"last"]
    dupli<-as.integer(unlist(dupli))

    allobjnames<-as.integer(c(remap$last,remap$new))
    
    
    lastobj <- max(allobjnames)
    
    watermasktmp<-watermask
    watermasktmp[]<-0
    
    done<-c()
    for (pos in 1:nrow(remap)) {            #permutate names
      old <- as.integer(remap[pos, "last"])
      new <- as.integer(remap[pos, "new"])
      dist<- as.double(remap[pos, "dist"])
      
      
      if( (old %in% dupli)&(old %in% done)&(dist>thr) ){

        done<-c(done,old)
        lastobj <- max(allobjnames)+1000

        allobjnames<-c(allobjnames,lastobj)
        
        old<-lastobj
        
      }else{
        done<-c(done,old)
      }
      
      
      watermasktmp[which(watermask==new)]<-old
      
    }
    
    watermask<-watermasktmp
    
    
    #####################compute center of objects for tracking##################################
    getobjcenter(watermask,1)
    
    
    switch<-0
    lastposition<-lastposition[0,]
    
    
  }
  
  
  #save nuclei mask as text file
  write.table(watermask, paste(resultnucleipath,testimage_file,".txt",sep=""), row.names = FALSE, col.names = FALSE)
  
  
  gaps<-array(data=1,dim = c(input_sizeX,input_sizeY))
  gaps[which(predtmp[,,1]==0 & predtmp[,,2]==0)]<-0
  
  # Voronoilambda<-100
  voronoiExamp = propagate(seeds = watermask, x = predtmp[,,1], mask = gaps, lambda = Voronoilambda) #, lambda = 100

  
  if(extractnucfromcyto==TRUE){
    voronoiExamp[watermask>0]<-0
  }
  
  if(cytocolorlabel==TRUE){
    compositevoronoy<-image_composite(test,image_read(colorLabels(voronoiExamp)),operator = "blend",compose_args = cytoblend)
  }else{

    img1<-array(0, dim=c(input_sizeX,input_sizeY,3))
    img1[,,1]<-voronoiExamp
    img1<-image_read(img1)
    img1<-image_rotate(img1,90) #for color mask
    img1<-image_flop(img1)      #for color mask
    img1<-image_transparent(img1, "#000000", fuzz = 10)
    compositevoronoy<-image_composite(test,img1,operator = "blend",compose_args = cytoblend)
  }
  
  if(nuccolorlabel==TRUE){
    compositevoronoy<-image_composite(compositevoronoy, image_read(colorLabels(watermask)),operator = "blend",compose_args = nucblend)
  }else{

    img2<-array(0, dim=c(input_sizeX,input_sizeY,3))
    img2[,,2]<-watermask
    img2<-image_read(img2)
    img2<-image_rotate(img2,90) #for color mask
    img2<-image_flop(img2)      #for color mask
    img2<-image_transparent(img2, "#000000", fuzz = 10)
    compositevoronoy<-image_composite(compositevoronoy, img2,operator = "blend",compose_args = nucblend)
  }
  
  
  segmented = paintObjects(channel(voronoiExamp, "grey"), as_EBImage(compositevoronoy), col='#ffff00')  #add borders in yellow (#ffff00) or blue (#00C1FF)

  
  
  if(cytocolorlabel==TRUE){
    segmented<-image_composite(image_read(segmented), image_read(colorLabels (voronoiExamp)),operator = "blend",compose_args = cytoblend)
  }else{
    segmented<-image_composite(image_read(segmented), img1, operator = "blend",compose_args = cytoblend)
  }
  
  

  
  
  #save cytoplasm mask as text file
  write.table(voronoiExamp, paste(resultcytoplasmpath,testimage_file,".txt",sep=""), row.names = FALSE, col.names = FALSE)
}




#####################################################################################################
#####################################################################################################
#####################################################################################################




autocontrast<-function(img){
  
  #merge the layers within the TIF file
  layers<-0
  if(is.na(dim(img)[3]) == FALSE)
    layers<-dim(img)[3]
  if(layers>1){
    alllayers<-list()
    for(i in 1:layers){
      alllayers[[i]]<-img[,,i]
    }
    img<-do.call(pmax,alllayers)  #blend pics and keep max intensity pixels
  }
  

  

  max1 <-mean(img)*2
  min1 <-min(img)
  
  img<- (img-min1)/as.double(max1-min1)
  img[img>1]<-1
  
  
  return(img)
}



      nucblend<-40
      nucmaskthreshold<-0.5
      watersize1<-5
      watersize2<-500
      nuccolorlabel<-FALSE
      cytoblend<-40
      cytomaskthreshold<-0.3
      Voronoilambda<-100
      cytocolorlabel<-FALSE
      dobackground<-TRUE
      watersizegap1<-2
      watersizegap2<-250
      extractnucfromcyto<-TRUE
      

    #reset tracking for new movie
    if (file.exists("positions.txt")){file.remove("positions.txt")}
    if (file.exists("positions0.txt")){file.remove("positions0.txt")}
    
    biometry<<-biometry[0,]
    
    
    
    
    
    ####################################for parallel computing#######################
    
    
    
    #folder to save results into

    dir.create(targetmovie) #create dir if not done already
    setwd(targetmovie)
    
    
    
    alldir<-list.dirs(path = movieroot, full.names = F)
    alldir<-alldir[alldir != ""]
    alldir<-alldir[alldir != "--190306"]
    

    foreach (dir = alldir, .packages=c("keras","magick","shotGroups","EBImage","scales","dplyr","gtools","stringr") ) %dopar% {      #for parallel processing (1 thread by folder)  #r<-
      
      
      modelnuc <-get_unet_1024()
      modelcyto<-get_unet_1024()
      load_model_weights_hdf5(modelnuc ,unetpathnuc)
      load_model_weights_hdf5(modelcyto,unetpathcyto)
      

      # #!!!!!!!!!!!TMP for crash!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # if(dir=="190306")
      #   next()
      # #!!!!!!!!!!!TMP for crash!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
      
      
      targetdir<-paste0(targetmovie,dir,"/")
      dir.create( targetdir )
      
      
      
      
      #update paths for result files
      resultcompositepath <-paste0(targetmovie,"/",dir,"/results-composite/")
      resultmaskspath     <-paste0(targetmovie,"/",dir,"/results-masks/")
      resultwatershedpath <-paste0(targetmovie,"/",dir,"/results-watershed/")
      resultvoronoipath   <-paste0(targetmovie,"/",dir,"/results-voronoi/")
      resultfullannotpath <-paste0(targetmovie,"/",dir,"/results-annot/")
      resultnucleipath    <-paste0(targetmovie,"/",dir,"/results-nuclei/")
      resultcytoplasmpath <-paste0(targetmovie,"/",dir,"/results-cytoplasm/")
      redpath   <-paste0(targetmovie,"/",dir,"/red/")
      greenpath <-paste0(targetmovie,"/",dir,"/green/")
      dir.create( resultcompositepath)
      dir.create( resultmaskspath)
      dir.create( resultwatershedpath)
      dir.create( resultvoronoipath)
      dir.create( resultfullannotpath)
      dir.create( resultnucleipath)
      dir.create( resultcytoplasmpath)
      dir.create(redpath)
      dir.create(greenpath)
      
      
setwd(paste0(targetmovie,"/",dir))
      
      dir<-paste0(movieroot,dir)
      
      allfiles<-list.files(dir, full.names = F, pattern = "*.TIF")
      
      
      
# ############to keep only files corresponding to a csv coordinate file#######################
#       # dir<-"Z:/common/ProjectGeneArray/Movies/PolII/190418"
#       allcsvfiles<-list.files(dir, full.names = F, pattern = "*.csv")
#       allcsvfiles<-tools::file_path_sans_ext(basename(allcsvfiles)) #remove extension
#       alls<-str_extract(allcsvfiles,"_s[0-9]+")
#       allcsvfiles<-str_remove(allcsvfiles,alls)
#       filters<-paste( paste0(allcsvfiles,"_.+",alls,"_t[0-9]+.TIF", collapse="|") )
#       
#       allfiles<-allfiles[ str_detect(allfiles, filters )]
# ############################################################################################
      
      
      
      allbasename<-unique( gsub("(.+)_w([1-9].+)_s([0-9]+)_t([0-9]+).TIF$", "\\1", allfiles) )
      for(basename in allbasename){
        
        files<-list.files(dir, full.names = TRUE, pattern = paste0("^",basename,"_.+TIF$") )
        allmovie<-sort(as.integer(unique( gsub("(.+)_w([1-9].+)_s([0-9]+)_t([0-9]+).TIF$", "\\3", files) )))
        allframe<-sort(as.integer(unique( gsub("(.+)_w([1-9].+)_s([0-9]+)_t([0-9]+).TIF$", "\\4", files) )))
        allcolor<-sort(unique(                 gsub("(.+)_w([1-9].+)_s([0-9]+)_t([0-9]+).TIF$", "\\2", files) ))
        
        for(movie in allmovie){
          
          #reset tracking for new movie
          if (file.exists("positions.txt")){file.remove("positions.txt")}
          if (file.exists("positions0.txt")){file.remove("positions0.txt")}
          biometry<<-biometry[0,]
          
          
          countframe<-0
          for(frame in allframe){
              
              countframe<-countframe+1
                
              # movie<-"1"
              # frame<-"10"
              
              if(grepl("561",allcolor[1]) ){  #if red color is used first  (green=488)
                color1<-allcolor[2]
                color2<-allcolor[1]   #assuming there are only 2 colors
              }else{
                color1<-allcolor[1]
                color2<-allcolor[2]   #assuming there are only 2 colors
              }
              
              file1<-paste0(dir,"/",basename,"_w",color1,"_s",movie,"_t",frame,".TIF")
              file2<-paste0(dir,"/",basename,"_w",color2,"_s",movie,"_t",frame,".TIF")
              
              
              if(!file.exists(file1))
                next()
              if(!file.exists(file2))
                next()
              
              
              filename<-paste0(basename,"_s",movie,"_t",frame,".jpg")     #removed lazer color
              filepath<-paste0(targetdir,filename)   #write

              
              
# if(file.exists(filepath)) #for crash!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   next()
              
              
              file3<-paste0(targetdir,"/",basename,"_s",movie,"_BIOMETRY",".txt")  #removed color name
              
              img1<-readImage(file1,all = TRUE)
              img1<-autocontrast(img1)
              
              img2<-readImage(file2,all = TRUE)
              img2<-autocontrast(img2)
              

              
              cells = rgbImage(green=img1, red=img2,)  #mix channels with custom colors

              
              
              
              
              writeImage( cells,filepath,type = "jpeg",quality = 100 )
              writeImage( img1, paste0(targetdir,"green/",filename)  ,type = "jpeg",quality = 100 )
              writeImage( img2, paste0(targetdir,"red/",filename),type = "jpeg",quality = 100 )   
              
              
              #################
              #save the corresponding files to analyze the original files with spoton.r
              redgreenfiles<-paste0(filename," ",file1," ",file2)  #file green red
              write.table(redgreenfiles, file=paste0(targetdir,"redgreenfiles.txt"), append = TRUE, row.names = F,col.names = F,quote = F)
              #################
              
              
              print(paste0("processing ",filepath))

              createmask(filepath,filename,Voronoilambda,watersize1,watersize2,watersizegap1,watersizegap2,nucmaskthreshold,cytomaskthreshold,nucblend,cytoblend,nuccolorlabel,cytocolorlabel,extractnucfromcyto,dobackground)

              
          }

          #save biometry data
          write.table(biometry,file3, row.names = FALSE, col.names = TRUE, append = FALSE)
          
          
        }
        allcolor<-c()
        allmovie<-c()
        allframe<-c()
      }
      allbasename<-c()
      allfiles<-c()
      
    }
    

    stopCluster(cl)

print("all done!!")

