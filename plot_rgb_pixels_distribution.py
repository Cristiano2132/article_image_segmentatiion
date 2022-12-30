import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
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
    df_pixels = pd.read_csv(path_data, index_col=0)
    df_pixels = df_pixels.assign(MGVRI=lambda x: (
        x.green ** 2 - x.red ** 2) / (x.green ** 2 + x.red ** 2))
    df_pixels = df_pixels.assign(GLI=lambda x: (
        2 * x.green - x.red - x.blue) / (2 * x.green + x.red + x.blue))
    df_pixels = df_pixels.assign(MPRI=lambda x: (
        x.green - x.red) / (x.green + x.red))
    df_pixels = df_pixels.assign(RGVBI=lambda x: (
        x.green - x.red * x.blue) / (x.green ** 2 + x.red * x.blue))
    df_pixels = df_pixels.assign(ExG=lambda x: (
        2 * x.green - x.red - x.blue) / (x.green + x.red + x.blue))
    df_pixels = df_pixels.assign(VEG=lambda x: (x.green) /
                            (x.red ** 0.667 + x.blue ** (1 - 0.667)))
    df_pixels.replace({'target': {'Solo': 'Soil', 'Vegetacao': 'Vegetation' }},  inplace=True)


    # def normalize(x, x_min:float=0, x_max:float=255):
    #     return (x-x_min)/(x_max-x_min)

    # df_pixels[['red', 'blue', 'green']]= df_pixels[['red', 'blue', 'green']].apply(normalize)
    # df_pixels[['MGVRI', 'GLI']] = df_pixels[['MGVRI', 'GLI']].apply(lambda x: normalize(x, x_min=-1, x_max=1))
 
    column_names = df_pixels.drop(['target', 'x', 'y'], axis=1).columns
    data =  df_pixels.drop(['target', 'x', 'y'], axis=1)
    # perform a robust scaler transform of the dataset
    trans = MinMaxScaler()
    data = trans.fit_transform(data)
    # convert the array back to a dataframe
    dataset = pd.DataFrame(data=data, columns=column_names)
    dataset['target']=df_pixels.target
 





    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    pal= [ "brown", "green"]
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    vars = ['red', 'blue', 'green', 'MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG']
    for ax, var  in zip(axs, vars):
        gfg = sns.kdeplot(data=dataset, x=var, hue='target',palette=pal,shade=True, ax=ax)
        gfg.legend_.set_title(None)
        plt.setp(gfg.get_legend().get_texts(), fontsize='12')
        ax.legend_.set_bbox_to_anchor((0.0, 1))
        ax.legend_._set_loc(2)
    plt.tight_layout()
    plt.show()
    fig.savefig('histogram_rgb_and_indices.png', dpi=100)

    z_dict = {'vars':[], 'z_values':[]}
    for n,var in enumerate(df_pixels.drop(labels=['x', 'y', 'target'], axis=1).columns):
        population_1 = 'Soil'
        population_2 = 'Vegetation'
        p1 = df_pixels.query(f"target=='{population_1}'")[[var]]
        p2 = df_pixels.query(f"target=='{population_2}'")[[var]]
        std_1 = p1.std()[0]
        std_2 = p2.std()[0]
        m_1 = p1.mean()[0]
        m_2 = p2.mean()[0]
        n_1 = len(p1)
        n_2 = len(p2)
        z = abs((m_1 - m_2)/sqrt(std_1**2/n_1 + std_2**2/n_2))
        z_dict['vars'].append(var)
        z_dict['z_values'].append(z)
    print(pd.DataFrame(z_dict).sort_values(by='z_values', axis=0, ascending=False).round(2))