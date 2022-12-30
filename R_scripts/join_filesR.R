# Lista de pacotes que serão utilizados:
r_pkgs <- c("tidyverse")

# Instala o pacote pacman se ele não estiver instalado:
if(!"pacman" %in% rownames(installed.packages())){
  install.packages("pacman")
} 

# Carrega os pacotes listados:
pacman::p_load(char = r_pkgs)
temp_df = df[FALSE,]
file_dir = 'data//output//'
for (f in list.files(file_dir)){
  file_path = paste(file_dir, f, sep="")
  df = read.csv(file_path, header = TRUE, row.names = 1)
  temp_df = temp_df %>% rbind(df)
}

temp_df %>% write.csv(file = "data//output//all_targets.csv")

