import torch 

sys.path.append("/home/yerinyoon/code/anonymousNet/data/celeba/")
## Load the trained generator.
self.restore_model(self.test_iters)
      
      # Set data loader.
if self.dataset == 'CelebA':
    data_loader = self.celeba_loader
elif self.dataset == 'RaFD':
    data_loader = self.rafd_loader
      
    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):

              # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg= self.create_labels(c_org, self.c_dim, self.dataset,self.selected_attrs, self.service_attrs)

              # Translate 
            x_fake_list = [x_real]
            x_fake_list.append(self.G(x_real, c_trg))
                #

              # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-skin-fixed-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))
              
  # Load the trained generator.
    self.restore_model(self.test_iters)
      
      # Set data loader.
    if self.dataset == 'CelebA':
        data_loader = self.celeba_loader
    elif self.dataset == 'RaFD':
        data_loader = self.rafd_loader
      
    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):

              # Prepare input images and target domain labels.
              x_real = x_real.to(self.device)
              c_trg= self.create_labels(c_org, self.c_dim, self.dataset,self.selected_attrs, self.service_attrs)

              # Translate 
              x_fake_list = [x_real]
              x_fake_list.append(self.G(x_real, c_trg))
                #

              # Save the translated images.
              x_concat = torch.cat(x_fake_list, dim=3)
              result_path = os.path.join(self.result_dir, '{}-skin-fixed-images.jpg'.format(i+1))
              save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
              print('Saved real and fake images into {}...'.format(result_path))
              
  
