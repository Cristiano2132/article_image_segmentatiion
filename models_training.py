from model_training import ModelsTraining
import matplotlib.pyplot as plt


plt.style.use('seaborn-poster')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['patch.edgecolor'] = '#ffffff'
plt.rcParams['patch.facecolor'] = '#ffffff'
plt.rcParams['savefig.edgecolor'] = '#ffffff'
plt.rcParams['savefig.facecolor'] = '#ffffff'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
# treinamento dos modelos
if __name__ == '__main__':
    random_state = 1
    path_data = 'data/output/all_targets.csv'
    models_rgb = ModelsTraining(data_path=path_data, save_folder_path='data/output/rgb', get_indices=False)
    df_rgb = models_rgb.trainin_models()
    df_rgb.rename({'index': 'target'}, axis=1, inplace=True)
    df_rgb.query("target in ['0', '1']", inplace=True)


    models_indices = ModelsTraining(data_path=path_data, save_folder_path='data/output/rgb_indices', get_indices=True)
    df_indices = models_indices.trainin_models()
    df_indices.rename({'index': 'target'}, axis=1, inplace=True)
    df_indices.query("target in ['0', '1']", inplace=True)
    df_rgb['variables']='rgb'
    df_indices['variables']='Indices'
    df_indices = df_indices.append(df_rgb, ignore_index=True).sort_values(['variables', 'target', 'model']).round(3)
    # df_indices.to_csv('resultados.csv')



    