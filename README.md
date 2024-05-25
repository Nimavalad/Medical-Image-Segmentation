# Medical-Image-Segmentation
def padimages(image,file_name, ratio):
    [length, width] = np.shape(image)
    if length/width>ratio:#1024/800
        print('This image needs padding.')
        add_wid = round(length*(1/ratio)-width)
        pad = np.zeros((length,add_wid))
        pad = pad.astype(image.dtype)
        if '_R_' in file_name:
        #                pad on the left
            pad_image = np.concatenate((pad,image),axis=1)
        else:
            pad_image = np.concatenate((image,pad),axis=1)
    return pad_image
#from skimage import data, color, io, img_as_float
image_path = "/content/inbreast-dataset/INbreast Release 1.0/AllPNG/"
annotation_path = '/content/inbreast-dataset/INbreast Release 1.0/AllXML/'
save_image_path = "/preprocessed_image/"
if not os.path.exists(save_image_path):
    os.mkdir(save_image_path)
save_mask_path = "/preprocessed_mask/"
if not os.path.exists(save_mask_path):
    os.mkdir(save_mask_path)
file_names = os.listdir(image_path)
file_names = sorted(file_names)
for i in range(0,len(file_names)):
    print(file_names[i])
    mammo = io.imread(image_path+file_names[i],0)
    xml_file_path = annotation_path + file_names[i].replace('png', 'xml')
    if not os.path.exists(xml_file_path):
        print(f"XML file not found for {file_names[i]}. Skipping this image.")
        continue
    lesion_mask = load_inbreast_mask(xml_file_path, mammo.shape[0:2])
    if np.max(lesion_mask)>=0:
    #Extract the breast profile and crop the mammogram, breast mask and the lesion mask
    #Normalize the image into 16-bit
     breast_preprocess = Preprocess.extract_breast_profile(mammo,lesion_mask,1)
    mammo = breast_preprocess.image
    breast_mask = breast_preprocess.mask
    lesion_mask =breast_preprocess.lesion_mask
    print ('Number of lesions: '+str(np.max(np.unique(label(lesion_mask)))))
    #pad the image, to ensure the aspect ratio is 1:1
    pad_mammo = padimages(mammo,file_names[i],1)
    #save the preprocessed image
    io.imsave(save_image_path + file_names[i],pad_mammo)
    #if the image has more than 1 lesion, then seperate them into different masks and number them.
    labelim = label(lesion_mask)
    if np.max(labelim)>0:
    #if there is at least 1 lesion.
        for l in range(1,np.max(labelim+1)):
          l_mask = np.zeros(np.shape(labelim))
          l_mask = l_mask.astype(lesion_mask.dtype)
          l_mask [labelim==l] = 255
          num_nonzero = np.where(l_mask>0)
          num_nonzero = len(num_nonzero[0])
          if num_nonzero>15:
              print('A valid mask')
              #Pad the mask in the same way as padding the image
              pad_l_mask = padimages(l_mask,file_names[i],1)
              io.imsave(save_mask_path+file_names[i][:-4]+str(l)+'.png',pad_l_mask)
          else:
              print('Has a tiny piece of noise that is not valid for training!')
    else:# if there is no lesion
          pad_lesion_mask = padimages(lesion_mask,file_names[i],1)
          io.imsave(save_mask_path+file_names[i][:-4]+str(0)+'.png',pad_lesion_mask)
stop = timeit.default_timer()
print('RunTime per image: ', (stop - start)/ len(file_names))
