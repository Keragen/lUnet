
options(error=traceback)
options(show.error.locations = TRUE)

##########################################################################################################################
#analyzes signals to measure intensities (using lunet masks and csv position files), create graph etc.                   #
#also generate mask files from spots for later machine learning (to clean these masks and remove what is not true signal)#
##########################################################################################################################

library(magick)
library(EBImage)
library(ggplot2)
#library(DescTools) #to check points in polygon
library(sp)
library(keras)
library(stringr)
library(geometry)   #concave hull

library(foreach)    #for parallel processing
library(doParallel) #for mac and linux


imgfileext<-"jpg"
#input_size<-1024
input_sizeX<-2048   #do not change. it will adapt automatically
input_sizeY<-2048   #do not change. it will adapt automatically
windowsize <-40     #do not change. it will adapt automatically
thrmethod<-"white"

Xfactor<-2048
Yfactor<-2048



##############to run on slurm instead of local for Atilla##############
modelspotpath<-"/shared/space2/projects/genearray/tools/fullmodelSpot.h5"     

setwd('/shared/space2/projects/genearray/Movies/')                            #folder with original movies

spotpath<-  '/shared/space2/projects/genearray/tools/spotpics'                #folder where the analysis is saved into
targetpath<-'/shared/space2/projects/genearray/tools/moviesUnet'              #pics to analyze (generated bu LUnetserver)
AIspotpath<-"/shared/space2/projects/genearray/tools/AIspots/pics/"           #to save signal samples for machine learning
###


##########################to set up#############################
alldir0<-c("AO2022-21-TEST")
finalMaskpath<-"finalMask-AO2022-21-TEST"


# tmp out file
#fileConn<-"/shared/space2/projects/genearray/tools/spotpics/AO2022-21-TEST/sample1/output.txt"
#cat("HELLO", file=fileConn,sep="\n",append=TRUE)
################################################################



finalMaskpath2<-paste0(AIspotpath,finalMaskpath)   #to save spot masks
if(! dir.exists(finalMaskpath2))
  dir.create(finalMaskpath2)


finalNUCMaskpath<-"NUCmask"
finalNUCMaskpath2<-paste0(AIspotpath,finalMaskpath,"/",finalNUCMaskpath)  #to save nuclear masks
if(! dir.exists(finalNUCMaskpath2))
  dir.create(finalNUCMaskpath2)



registerDoSEQ()
cl <- makeCluster(30, outfile="out.txt")  #set number of threads (set to match max number of tracks in csv file)
registerDoParallel(cl)

##############functions#####################################################



# #############################################
# ###############for test######################
# setwd('C:/Users/polo/Desktop/Attila/')                               #folder with original movies
# spotpath<-  'C:/Users/polo/Desktop/Attila/spotpics'                  #folder where the analysis is saved into
# targetpath<-'C:/Users/polo/Desktop/Attila/targetmovies'              #pics to analyze (generated bu LUnetserver)
# AIspotpath<-"/shared/space2/projects/genearray/tools/AIspots/pics/"  #to save signal samples for machine learning
# alldir0<-c("test")
# #############################################
# #############################################





autocontrast<-function(img){
  
  max1 <-mean(img)*2
  min1 <-min(img)
  
  img<- (img-min1)/as.double(max1-min1)
  img[img>1]<-1
  
  return(img)
}

testimageRead <- function(image_file) {
  img <- image_read(image_file)
  return(img)
}



getspotmask <- function(spot,method,limitmask,filename,lastspotcenter,loop=0){
  
  
  truespot<-spot #keep the real thing for machine learning
  
  spot_th<-spot
  spot_th<-spot_th[limitmask>0]  #threshold only in nucleus
  
  
  meanspot<-mean(spot_th)
  sdspot<-sd(spot_th)
  thrspot<-meanspot+(2*sdspot)
  
  spot_th<-spot
  spot_th[spot<thrspot]<-0
  
  #spot_th[limitmask==0]<-0  #remove signal not in nucleus  #test a remettre si plusieurs spots!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  spot_th[spot_th>0]<-1
  
  
  

  # ############
  nmask<-watershed( distmap(spot_th), 2 )
  minsurface<-4
  allobj<-table(nmask@.Data)  #each number correspond to a distinct object
  toremove<-allobj[allobj<minsurface]
  nmask = rmObjects(nmask, names(toremove), reenumerate=TRUE)
  # ############
  
  #to remove small gaps inside masks
  nmask<-fillHull(nmask)
  

  
  #save separate spots for later machine learning
  if(loop==0){ #?  save only for first round of spot detection
    if(filename!="toto"){ #save spots only for red channel
      allobj<-table(nmask@.Data)
      for(obj in names(allobj)){
        
        if(obj=="0")
          next()
        
        allobj2<-allobj[! names(allobj)==obj]
        tmpobj<-rmObjects(nmask, names(allobj2), reenumerate=FALSE)
        
        objpics<-as.Image( as.matrix(tmpobj) ) #to save b&w mask
        pathtosave<-paste0(AIspotpath,filename,"-obj",obj,".jpg")
        
      }
    }
  }
  
  
  

  #use machine learning to delete fake signals
  if(filename!="toto"){ #correct spots only for red channel
    removed<-0
    allobj<-table(nmask@.Data)
    for(obj in names(allobj)){
      
      if(obj=="0")
        next()
      
      allobj2<-allobj[! names(allobj)==obj]
      tmpobj<-rmObjects(nmask, names(allobj2), reenumerate=FALSE)
      
      
      ###convert to pic and reduce to 20x20 in order to use machine learning prediction!!!###############
      tmpobj[tmpobj>0]<-1
      objpics<-as.Image( as.matrix(tmpobj) ) #   !!!
      objpics<-resize(objpics, 20,20)#   !!!
      ###################################################################################################
      
      
      #tmpobj<-as.matrix(tmpobj)*255   !!!
      tmpobj<-imageData(objpics)*255
      
      
      tmpobj<-array_reshape(tmpobj,c(-1,20,20,1))       #!!!
      
      
      
      result<- predict(modelspot,tmpobj)
      if(result<0.6){ #threshold of 60% from prediction
        
        limitmask[ nmask==obj ]<-0
        
        nmask<-rmObjects(nmask, obj, reenumerate=FALSE)
        removed<-1
      }
      
    }
    
    #################  
    
    ####repeat the thresholding in case of problem
    allobj<-table(nmask@.Data)
    if(length(allobj[names(allobj)!="0"])==0 & removed==1){ #if mask is blank because of bad signal, do the thresholding again
      loop<-loop+1

      if(loop==1){
        
        nmask<-getspotmask(spot,method,limitmask,filename,lastspotcenter,loop) #1 is to limit to 1 loop
        
      }
    }
    
    assign("masknucleus2", limitmask, envir = .GlobalEnv) #save corrected nucleus mask
    
  }
  
  nmask[nmask>0]<-1 #set everything to 1
  
  
  
  
  ###check if we still have several objects in the mask and keep those that overlap with the previous spot
  if(lastspotcenter!=""){
    if(any(nmask>0)){
      allobj<-watershed( distmap(nmask), 2 )
      
      oldM<-which(lastspotcenter>0)
      for(eachspot in names(table(allobj)) ){
        if(eachspot==0)
          next()
        
        newM<-which(allobj==eachspot)
        
        
        overlapixels<-sum(unique(oldM) %in% unique(newM))
        
        if(overlapixels==0){
          allobj<-rmObjects(allobj, eachspot, reenumerate=FALSE)
        }
        
      }
      nmask<-allobj
      nmask[nmask>0]<-1 #set everything to 1
    }
  }
  
  return(nmask)
}



maxprojection<-function(img){
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
  return(img)
}
##########end of functions#########################################################




#create file to save data
datafile<-paste(spotpath,"/signal_data_",paste(alldir0,collapse="_"),".txt",sep="")   #paste0(spotpath,"/signal_data.txt")
datafileoriginal<-paste(spotpath,"/signal_data_original_",paste(alldir0,collapse="_"),".txt",sep="")

if(file.exists(datafile))
  unlink(datafile)
file.create(datafile)

if(file.exists(datafileoriginal))
  unlink(datafileoriginal)
file.create(datafileoriginal)


for(nextdir0 in alldir0){
  alldir<-list.dirs(nextdir0,F,F)    #eg PolII
  
  
  print(alldir) #tmp
  
  
  newdir<-paste0(spotpath,"/",nextdir0)
  if(! dir.exists(newdir))
    dir.create(newdir)
  

  for(nextdir in alldir){
    
    allTIFFfiles<-list.files(paste0("./",nextdir0,"/",nextdir), full.names = T, pattern = "*.TIF")
    allcsvfiles<-list.files(paste0("./",nextdir0,"/",nextdir), full.names = T, pattern = "*.csv")
    
cat(allcsvfiles, file=fileConn,sep="\n",append=TRUE)

    if(! dir.exists(paste0(targetpath,"/",nextdir0,"/",nextdir))){      next()    }
    if(! file.exists(paste0(targetpath,"/",nextdir0,"/",nextdir,"/","redgreenfiles.txt"))){      next()    }
    
    originalredgreenfiles<-read.table(paste0(targetpath,"/",nextdir0,"/",nextdir,"/","redgreenfiles.txt"),header = F,col.names = c("file", "green", "red"),stringsAsFactors = F,sep = " ")
    originalredgreenfiles<-unique(originalredgreenfiles) #not tested
    
    newdir<-paste0(spotpath,"/",nextdir0,"/",nextdir)
    if(! dir.exists(newdir))
      dir.create(newdir)
    

    for(nextcsv in allcsvfiles){
      
      filecorename<-tools::file_path_sans_ext(basename(nextcsv))
      csvfilepath<-nextcsv
      print(paste0("processing ",nextdir0,"/",nextdir,"/",filecorename) )
      
      
cat(paste0("processing ",nextdir0,"/",nextdir,"/",filecorename), file=fileConn,sep="\n",append=TRUE)
      
      
      print (paste0("opening ",csvfilepath))
      
cat(paste0("csvfilepath: ",csvfilepath), file=fileConn,sep="\n",append=TRUE)
      
      #new format from Attila
      coords<-read.csv(file = csvfilepath, header = T, sep = "\t", col.names = c("row","track","frame","x","y","z") )  #fileEncoding = "UCS-2LE"
      names(coords)<-c("row","track","frame","x","y","z")
      
cat(paste0("coords: ",csvfilepath), file=fileConn,sep="\n",append=TRUE)    
      
      
      #fill the blanks in the csv file (frames start and stop when signal is present only. we need all the frames)
      maxallframe<-max(coords$frame)
      alltracks<-sort(unique(coords$track))
      newcoords<-coords[FALSE,]
      for(tr in alltracks){
        
        tmpcoords<-coords[coords$track == tr,]
        minframe<-min(tmpcoords$frame)
        maxframe<-max(tmpcoords$frame)
        
        minlist<-tmpcoords[tmpcoords$frame==minframe,]
        maxlist<-tmpcoords[tmpcoords$frame==maxframe,]
        
        
        if(minframe>1)
          for(uu in 1:(minframe-1)){
            minlist2<-minlist
            minlist2$frame<-uu
            newcoords<-rbind(newcoords,minlist2)
          }
        
        newcoords<-rbind(newcoords,tmpcoords)
        
        if(maxframe<maxallframe)
          for(uu in (maxframe+1):maxallframe){
            maxlist2<-maxlist
            maxlist2$frame<-uu
            newcoords<-rbind(newcoords,maxlist2)
          }
      }
      coords<-newcoords
      names(coords)<-c("row","track","frame","x","y","z")
      
      
      
      
      
      tracksplit<-split(coords, coords$track)
      
      
      
      
      foreach (coords = tracksplit, .packages=c("magick","EBImage","ggplot2","sp","keras","stringr","geometry", "foreach", "doParallel") ) %dopar% {      #for parallel processing (1 thread per track)
        
        
        modelspot<-load_model_hdf5(modelspotpath, compile = F)
        
        
        # coords
        #dataframe to save data from each csv file
        alldata<-data.frame(file=character(),
                            track=integer(),
                            frame=integer(),
                            Signal_Spot_Red=double(),
                            Signal_Spot_Green=double(),
                            Signal_Window_Red=double(),
                            Signal_Window_Green=double(),
                            Signal_BG_red=double(),
                            Signal_BG_green=double(),
                            Spot_Red_Size=double(),
                            Spot_Green_Size=double(),
                            BG_red_Size=double(),
                            BG_green_Size=double(),
                            Full_Red_Signal_Nucleus=double(),
                            Full_Red_Signal_Cytoplasm=double(),
                            Full_Green_Signal_Nucleus=double(),
                            Full_Green_Signal_Cytoplasm=double(),
                            Full_Nucleus_Size=double(),
                            Full_Cytoplasm_Size=double(),
                            stringsAsFactors = FALSE)
        
        alldataoriginal<-data.frame(file=character(),
                                    track=integer(),
                                    frame=integer(),
                                    Signal_Spot_Red=double(),
                                    Signal_Spot_Green=double(),
                                    Signal_Window_Red=double(),
                                    Signal_Window_Green=double(),
                                    Signal_BG_red=double(),
                                    Signal_BG_green=double(),
                                    Spot_Red_Size=double(),
                                    Spot_Green_Size=double(),
                                    BG_red_Size=double(),
                                    BG_green_Size=double(),
                                    Full_Red_Signal_Nucleus=double(),
                                    Full_Red_Signal_Cytoplasm=double(),
                                    Full_Green_Signal_Nucleus=double(),
                                    Full_Green_Signal_Cytoplasm=double(),
                                    Full_Nucleus_Size=double(),
                                    Full_Cytoplasm_Size=double(),
                                    stringsAsFactors = FALSE)
        
        
        
        

        
        
        r<-min(coords$frame) #to start from 1st frame in the csv file
        rmax<-max(coords$frame) #to start from 1st frame in the csv file
        
        lastspotcenter<-""
        
        for(r in nrow(coords):1){     # to go backward!!
          # r<-1  #first row
          frame<-coords[r,"frame"]
          x    <-coords[r,"x"]
          y    <-coords[r,"y"]
          z    <-coords[r,"z"]
          track<-coords[r,"track"]
          
          
          
          
          #read files and resize to focus on the same 20x20 window
          imgname<-paste0(filecorename,'_t',frame,".",imgfileext)
          imgname2<-paste0(filecorename,'_track',track,'_t',frame,".",imgfileext)  #to save composite picture
          imgname4<-paste0(nextdir0,"-",nextdir,"-",filecorename,'_track',track,'_t',frame)  #to save spots
          
          cat("processing: ",r," -- ",imgname,"\n")
          
          
          imgpath<-paste0(targetpath,"/",nextdir0,"/",nextdir,'/results-nuclei/',imgname,".txt")
          masknucleus<-read.table(imgpath,header = F,stringsAsFactors = F)
          
          
          imgpath<-paste0(targetpath,"/",nextdir0,"/",nextdir,'/results-cytoplasm/',imgname,".txt")
          maskcytoplasm<-read.table(imgpath,header = F,stringsAsFactors = F)
          
          
          
          
          ####adapt tools to picture resolution######
          #dimension of Unet mask
          input_sizeX<-dim(masknucleus)[1]  #!!!
          input_sizeY<-dim(masknucleus)[2]  #!!!
          
          if (Xfactor==2048){          #(input_sizeX==2048){  #dim fof original pic
            windowsize<-40
            
            if(input_sizeX==1024){  #1024fix                  #dimension of Unet mask
              masknucleus<-masknucleus[rep(1:nrow(masknucleus), each=2), rep(1:ncol(masknucleus), each=2)]         #1024fix  resize mask to fit picture
              maskcytoplasm<-maskcytoplasm[rep(1:nrow(maskcytoplasm), each=2), rep(1:ncol(maskcytoplasm), each=2)] #1024fix  resize mask to fit picture
            }
          }else{  #1024?
            windowsize<-20
          }
          x<-x-round(windowsize/2)  #!!!
          y<-y-round(windowsize/2)  #!!!
          
          
          if(x<1) x<-1
          if(y<1) y<-1
          if( (x+windowsize) > Xfactor){x<- Xfactor-windowsize}
          if( (y+windowsize) > Yfactor){y<- Yfactor-windowsize}
          ###########################################
          
          
          masknucleus  <-matrix(unlist(masknucleus), ncol =2048, nrow =2048)   #convert back to matrix  #lll
          maskcytoplasm<-matrix(unlist(maskcytoplasm), ncol =2048, nrow =2048) #convert back to matrix  #lll
          
          
          masknucleus2<-masknucleus[seq(x,x+windowsize-1),seq(y,y+windowsize-1)]
          
          
          nucname<-table(as.matrix(masknucleus2))         #get the name of the corresponding nucleus
          nucname<-names(nucname[nucname==max(nucname)])  #the largest surface get the name
          
          
          
          
          
          if(nucname == "0"){   #make sure a nucleus is detected next to the spot
            masknucleus2[]<-1   #when no nucleus around spot, set bg as nucleus to avoid a crash. remove when Unet works again
          }    
          
          
          #open original pictures
          originalfiles<-originalredgreenfiles[originalredgreenfiles[1]==imgname]
          imgreenoriginal<- readImage(originalfiles[2],all = TRUE) #green
          imredoriginal  <- readImage(originalfiles[3],all = TRUE) #red
          
          
          #get the corresponding Z
          imgreenoriginal<-imgreenoriginal[,,z]
          imredoriginal  <-imredoriginal[,,z]
          
          imgred  <-imredoriginal
          imggreen<-imgreenoriginal
          
          

          #create masks to remove bright spots#######################################
          seuilred<-quantile(imgred, c(.99))  #could be useful to detect gene arrays!
          seuilgreen<-quantile(imggreen, c(.99))

          imggreenMask<-imggreen
          imggreenMask[imggreenMask>seuilgreen]<-1
          imggreenMask<-medianFilter(imggreenMask, 2) # to remove speckles
          imggreenMask[imggreenMask<1]<-0
          

          imgredMask<-imgred
          imgredMask[imgredMask>seuilred]<-1
          imgredMask<-medianFilter(imgredMask, 2) # to remove speckles
          imgredMask[imgredMask<1]<-0

          
          #TMP!!!!!save masks for control!!!!!
          attimask<-as.Image( as.matrix(imggreenMask) ) #to save b&w mask
          pathtosave<-paste0("/shared/space2/projects/genearray/tools/AIspots/pics/finalMask-TEST/fullGREENmask/",imgname)
          writeImage(attimask, pathtosave,quality = 100)


          attimask<-as.Image( as.matrix(imgredMask) ) #to save b&w mask
          pathtosave<-paste0("/shared/space2/projects/genearray/tools/AIspots/pics/finalMask-TEST/fullREDmask/",imgname)
          writeImage(attimask, pathtosave,quality = 100)

          ##############################

          
          
          #mesure sur images originales sans contraste
          
          fullredsignalnucleus     <- 0   #reset
          fullredsignalcytoplasm   <- 0
          fullgreensignalnucleus   <- 0
          fullgreensignalcytoplasm <- 0
          fullnucleussize    <-   0
          fullcytoplasmsize  <-   0
          
          
          
          
          #NOT using which
          fullredsignalnucleus     <- mean(imgred[masknucleus==nucname])
          fullredsignalcytoplasm   <- mean(imgred[maskcytoplasm==nucname])

          fullgreensignalnucleus   <- mean(imggreen[Reduce(intersect, list(which(masknucleus==nucname),which(imggreenMask!=1)))])    #use mask to remove bright antibody clumps
          fullgreensignalcytoplasm <- mean(imggreen[Reduce(intersect, list(which(maskcytoplasm==nucname),which(imggreenMask!=1)))])

          fullnucleussize    <-   length(imgred[masknucleus==nucname])
          fullcytoplasmsize  <-   length(imgred[maskcytoplasm==nucname])
          
          
          
          #set contrast
          imgred  <- autocontrast(imredoriginal) #red
          imggreen<- autocontrast(imgreenoriginal)  #green
          
          #use CLAHE contrast
          imgred<-clahe(imgred, nx = 64, ny = 64, bins = 512, limit = 5, keep.range = FALSE)  #CLAHE
          imggreen<-clahe(imggreen, nx = 64, ny = 64, bins = 512, limit = 5, keep.range = FALSE)  #CLAHE
          
          
          
          #crop
          xi<-seq(x,x+windowsize-1)
          yi<-seq(y,y+windowsize-1)
          
          imredoriginal  <-imredoriginal[xi,yi]
          imgreenoriginal<-imgreenoriginal[xi,yi]
          
          imgred  <-imgred[xi,yi]
          imggreen<-imggreen[xi,yi]
          
          
          
          
          ######patch the nucleus mask with a chull to make a smooth patatoid######
          
          nuc<-ifelse(masknucleus2>0,1,0)
          nuc<-fillHull(nuc)
          nuc2<- nuc@.Data== 1
          nucpixcoord<-which(nuc2==TRUE, arr.ind=T)
          
          
          
          
          #convex hull
          H  <- chull(nucpixcoord)
          nucmask<-which(nuc2==nuc2, arr.ind=T)
          
          new_shape <- sp::point.in.polygon( nucmask[,1], nucmask[,2],nucpixcoord[H,1],nucpixcoord[H,2])
          new_shape<-nucmask[new_shape>0,]
          
          masknucleusconvex<-array(dim = c(windowsize,windowsize),0)
          masknucleusconvex[as.matrix(new_shape)]<-1
          
          
          masknucleus2<-masknucleusconvex
          
          
          spotmaskred<-getspotmask(imgred,thrmethod,masknucleus2,imgname4,lastspotcenter)   #detect signal using a thresholding method applied only on nucleus part
          
          
          ####get pixels coordinates from signal mask###
          
          
          spotmaskgreen<-getspotmask(imggreen,thrmethod,masknucleus2,"toto","") #lastspotcenter)
          
          if(any(spotmaskred>0)){
            lastspotcenter<-spotmaskred
          }
          
          
          #save mask for Attila
          attimask<-as.Image( as.matrix(spotmaskred) ) #to save b&w mask
          pathtosave<-paste0(AIspotpath,finalMaskpath,"/track",track,"-",imgname)
          writeImage(attimask, pathtosave,quality = 100)
          
          
          #save NUC mask
          attimask<-as.Image( as.matrix(masknucleus2) ) #to save b&w mask
          pathtosave<-paste0(AIspotpath,finalMaskpath,"/",finalNUCMaskpath,"/track",track,"-",imgname)
          writeImage(attimask, pathtosave,quality = 100)
          
          
          
          
          
          if(nucname == "0"){ #make sure a nucleus is detected next to the spot
            next()
          }
          
          masknucleus2[spotmaskred==1]  <-1 #add spot mask parts that are not in nucleus (false negative) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          
          
          
          
          ######################################
          #measurment on ORIGINAL picture files#
          ######################################
          #intensities are normalized between 0 to 1. to get the real number x 65535 (16bits)
          
          
          #NOT using which
          signalwindowred  <-sum( imredoriginal[masknucleus2>0] )      #intensity in the nucleus (in the frame) with the spot
          signalwindowgreen<-sum( imgreenoriginal[masknucleus2>0] )

          signalspotred  <-sum( imredoriginal[spotmaskred>0] )
          signalspotgreen<-sum( imgreenoriginal[spotmaskred>0] )

          signalbgred   <- mean( imredoriginal[spotmaskred==0 & masknucleus2>0] )      #intensity in the nucleus (in the frame) without the spot
          signalbggreen <- mean( imgreenoriginal[spotmaskred==0 & masknucleus2>0] )

          bgredsize     <- length(imredoriginal[spotmaskred==0 & masknucleus2>0])
          bggreensize   <- length(imgreenoriginal[spotmaskgreen==0 & masknucleus2>0]) #not used

          spotredsize   <- length( imredoriginal[spotmaskred>0] )
          spotgreensize <- length( imgreenoriginal[spotmaskgreen>0] ) #not used
          
          
          
          cat(csvfilepath," ",track, " ",frame," ",signalspotred," ",signalspotgreen," ",signalwindowred," ",signalwindowgreen," ",signalbgred," ",signalbggreen," ",spotredsize," ",spotgreensize," ",bgredsize," ",bggreensize," ",fullredsignalnucleus," ",fullredsignalcytoplasm," ",fullgreensignalnucleus," ",fullgreensignalcytoplasm," ",fullnucleussize," ",fullcytoplasmsize,"\n")
          alldata<-rbind(alldata,c(csvfilepath,track,frame,signalspotred,signalspotgreen,signalwindowred,signalwindowgreen,signalbgred,signalbggreen,spotredsize,spotgreensize,bgredsize,bggreensize,fullredsignalnucleus,fullredsignalcytoplasm,fullgreensignalnucleus,fullgreensignalcytoplasm,fullnucleussize,fullcytoplasmsize), stringsAsFactors = FALSE)
          names(alldata)<-c("file", "track", "frame", "Signal_Spot_Red", "Signal_Spot_Green", "Signal_Window_Red", "Signal_Window_Green", "Signal_BG_red", "Signal_BG_green", "Spot_Red_Size", "Spot_Green_Size", "BG_red_Size", "BG_green_Size", "Full_Red_Signal_Nucleus","Full_Red_Signal_Cytoplasm","Full_Green_Signal_Nucleus","Full_Green_Signal_Cytoplasm","Full_Nucleus_Size","Full_Cytoplasm_Size")
          
          
          ##################################################
          
          
          ####  make a colored composite with bg masks and spot
          imgmask<-magick::image_read(rgbImage(masknucleus2*0,masknucleus2*255,masknucleus2*0))
          
          
          spotmaskgreen<-rgbImage(spotmaskgreen*255,spotmaskgreen*255,spotmaskgreen*255)
          spotmaskred  <-rgbImage(spotmaskred*255,spotmaskred*255,spotmaskred*255)
          
          
          imgmaskgreen  <-image_transparent(magick::image_read(spotmaskgreen), "black", fuzz = 10)
          imgmaskgreen  <-image_composite(imgmask,imgmaskgreen,operator = "atop")
          imgmaskgreen<-as_EBImage(imgmaskgreen)
          
          imgmaskred  <-image_transparent(magick::image_read(spotmaskred), "black", fuzz = 10)
          imgmaskred  <-image_composite(imgmask,imgmaskred,operator = "atop")
          imgmaskred  <-as_EBImage(imgmaskred)
          ####
          
          
          
          #composite with wide window to check quality of green and red masks from AI##########################
          imgpath<-paste0(targetpath,"/",nextdir0,"/",nextdir,'/',imgname)   #warning this is the max proj!!!
          imgfull<-testimageRead(imgpath)
          imgfull<-image_crop(imgfull, geometry_area(windowsize*2,windowsize*2,x-windowsize/2,y-windowsize/2) )
          
          imgpath     <-paste0(targetpath,"/",nextdir0,"/",nextdir,'/results-masks/',imgname)
          imgmaskfull <-testimageRead(imgpath)
          
          ######tmp teletravail###
          if (Xfactor==2048){          #(input_sizeX==2048){  #dim fof original pic
            if(input_sizeX==1024){  #1024fix                  #dimension of Unet mask
              imgmaskfull<-image_scale(imgmaskfull, paste0(2048, "x", 2048, "!")) #1024fix  resize mask to fit picture
            }
          }
          

          #########################
          
          imgmaskfull <-image_crop(imgmaskfull, geometry_area(windowsize*2,windowsize*2,x-windowsize/2,y-windowsize/2) )
          

          imgmaskfull <-image_composite(imgfull,imgmaskfull,operator = "blend", compose_args =35)
          imgmaskfull <-as_EBImage(imgmaskfull)
          
          ####
          
          #save composite pictures to check the quality of predictions
          jpeg(paste0(spotpath,"/",nextdir0,"/",nextdir,"/",imgname2), quality = 100, width = 600, height = 1000, res = 96)
          layout(matrix(c(1,1,1,2,3,4,5,6,7), nrow = 3, ncol = 3, byrow = TRUE))
          plot(imgmaskfull)+rect(windowsize/2,windowsize/2,windowsize*2-windowsize/2,windowsize*2-windowsize/2,border = "yellow",lwd = 5)
          plot(imgred)
          plot(spotmaskred)
          plot(imgmaskred)
          plot(imggreen)
          plot(spotmaskgreen)
          plot(imgmaskgreen)
          dev.off()
          
          
        }
        
        
        
        # #measurment on original picture files##################
        #compute results and make graph for each track
        
        
        #new easy way
        alldata$signalred   <- as.numeric(alldata$Signal_Spot_Red)   - (as.numeric(alldata$Spot_Red_Size)* as.numeric(alldata$Signal_BG_red))
        alldata$signalgreen <- as.numeric(alldata$Signal_Spot_Green) - (as.numeric(alldata$Spot_Red_Size)* as.numeric(alldata$Signal_BG_green))  #Signal_Spot_Green is based on the red mask!!
        
         
        #!!doublon as is. maybe usefull later when multiplicated by the size
        alldata$fullnucleussignalred     <-  as.numeric(alldata$Full_Red_Signal_Nucleus)    
        alldata$fullnucleussignalgreen   <-  as.numeric(alldata$Full_Green_Signal_Nucleus)  
        alldata$fullcytoplasmsignalred   <-  as.numeric(alldata$Full_Red_Signal_Cytoplasm)  
        alldata$fullcytoplasmsignalgreen <-  as.numeric(alldata$Full_Green_Signal_Cytoplasm)
        
        
        alltracks<-unique(alldata$track)
        for(track in alltracks){

          data<-alldata[alldata$track==track,]
          
          imgname3<-paste0(filecorename,'_track',track,'_t',max(as.numeric(data$frame)),"-2.",imgfileext)  #to save graph
          jpeg(paste0(spotpath,"/",nextdir0,"/",nextdir,"/",imgname3), quality = 100, width = 600, height = 1000, res = 96)
          p = ggplot() + 
            geom_line(data = data, aes(x = as.numeric(frame), y = as.numeric(signalred)), color = "firebrick1") +
            geom_line(data = data, aes(x = as.numeric(frame), y = as.numeric(signalgreen)), color = "forestgreen") +
            
            stat_smooth(aes(x=as.numeric(data$frame), y=as.numeric(data$signalred)), method = lm, formula = y ~ poly(x, 6), se = FALSE, color=rgb(red = 1, green = 0.17, blue = 0.2, alpha = 0.2), linetype = "dashed") +
            stat_smooth(aes(x=as.numeric(data$frame), y=as.numeric(data$signalgreen)), method = lm, formula = y ~ poly(x, 6), se = FALSE, color=rgb(red = 0, green = 0.5, blue = 0.15, alpha = 0.2), linetype = "dashed") +
            xlab('Time') +
            ylab('Signal')
          
          print(p)
          dev.off()
          
          
          
          imgname5<-paste0(filecorename,'_track',track,'_t',max(as.numeric(data$frame)),"-3.",imgfileext)  #to save graph
          jpeg(paste0(spotpath,"/",nextdir0,"/",nextdir,"/",imgname5), quality = 100, width = 600, height = 1000, res = 96)
          p = ggplot() + 
            geom_line(data = data, aes(x = as.numeric(data$frame), y = as.numeric(data$fullnucleussignalred)  , colour = "Nucleus Red")) +
            geom_line(data = data, aes(x = as.numeric(data$frame), y = as.numeric(data$fullcytoplasmsignalred), colour = "Cytoplasm Red")) +
            geom_line(data = data, aes(x = as.numeric(data$frame), y = as.numeric(data$fullnucleussignalgreen), colour = "Nucleus Green")) +
            geom_line(data = data, aes(x = as.numeric(data$frame), y = as.numeric(data$fullcytoplasmsignalgreen), colour = "Cytoplasm Green")) +  #,linetype = "dashed"
            labs(x = "Time",
                 y = "Signal",
                 color = "Legend")
          
          
          print(p)
          dev.off()
          
          
        }
        
        ############################
        
        
        
        #save data in file
        write.table(alldata, datafile, append = TRUE, sep = "\t", dec = ".", row.names = F, col.names = T, na = "0")
      }#fin du foreach
    }#fin du allcsv
  }#fin du alldir (subfolders)
}#fin du alldir0 (main folders)

stopCluster(cl)
print("all done!!")

##the most useful way to look at many warnings:
summary(warnings())
