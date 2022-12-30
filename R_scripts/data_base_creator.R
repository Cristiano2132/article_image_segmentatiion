# Lista de pacotes que serão utilizados:
r_pkgs <- c("raster",
            "tidyverse",
            # "terra",
            "mapedit",
            "mapview",
            "sf")

# Instala o pacote pacman se ele não estiver instalado:
if(!"pacman" %in% rownames(installed.packages())){
  install.packages("pacman")
} 

# Carrega os pacotes listados:
pacman::p_load(char = r_pkgs)

# img_path = 'data//input//b1__219.2.tif_path'
# img_path = 'data//input//b1__223.2.tif_path'
# img_path = 'data//input//b1__224.2.tif_path'
# img_path = 'data//input//b1__234.2.tif_path'
# img_path = 'data//input//b1__244.2.tif_path'
img_path = 'data//output//test_mask.tif'
img <- brick(img_path) %>% `names<-`(c('red', 'green', 'blue', 'MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG', 'mask'))

img <- projectRaster(img, crs="+init=epsg:4326")


img[img==0] <- NaN
plotRGB(img,  axes = F, stretch = "hist", main = "True Color Composite")

# Recortando areas com vegetacao
pts_vegetacao <- viewRGB(img, 9) %>% editMap()
crop_vegetacao <- st_sf(pts_vegetacao$finished$geometry) 
img_vegetacao <- mask(img, crop_vegetacao)


## Imagem com vegetacao
df_vegetacao <- as(img_vegetacao, "SpatialPixelsDataFrame") %>% 
  as.data.frame() %>% 
  mutate(
    target = "Vegetacao"
  )


# Recortando areas com solo
# pts_solo <- viewRGB(img, r = 1, g = 2, b = 3) %>% editMap()
pts_solo <- viewRGB(img, 9) %>% editMap()
crop_solo <- st_sf(pts_solo$finished$geometry) 
img_solo <- mask(img, crop_solo)

df_solo <- as(img_solo, "SpatialPixelsDataFrame") %>% 
  as.data.frame() %>% 
  mutate(
    target = "Solo"
  )

df_full = df_solo %>% rbind(df_vegetacao)

df_full %>% write.csv(file = "data//output//targets_teste.csv")

