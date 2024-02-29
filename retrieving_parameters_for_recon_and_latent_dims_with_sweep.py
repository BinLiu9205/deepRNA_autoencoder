import pandas as pd
import json
import yaml

detail_file = pd.read_csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/model_and_parameter_information.csv", header=0)
yaml_config_base = "/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Post-hoc_and_visualization/results_and_metrics/conf/"
for i in range(71):
    if detail_file['Whether Sweep'][i] == 'Yes':
        json_acceptable_string = detail_file['Config'][i].replace("'", "\"")
        config_dict = json.loads(json_acceptable_string)
        runID = detail_file['Run ID'][i]
        if 'model' in config_dict:
            if 'model.encoder_config' in config_dict: 
                specific_config = {
                'latent_dim': config_dict['model']['latent_dim'],
                'encoder_config': config_dict['model.encoder_config'],
                'test_set': config_dict['datasets']['test_set'],
                'beta' : config_dict['training.beta'],
                'mu' : config_dict['gene_set_definition']['mu'],
                'sigma' : config_dict['gene_set_definition']['sigma'],
                'batch_size': config_dict['training.batch_size']
                }
            else:
                specific_config = {
                'latent_dim': config_dict['model']['latent_dim'],
                'encoder_config': config_dict['model']['encoder_config'],
                'test_set': config_dict['datasets']['test_set'],
                'beta' : config_dict['training.beta'],
                'mu' : config_dict['gene_set_definition']['mu'],
                'sigma' : config_dict['gene_set_definition']['sigma'],
                'batch_size': config_dict['training']['batch_size']
                }
        else:
            specific_config = {
            'latent_dim': 50,
            'encoder_config': [1000, 100],
            'test_set': '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl'
            }
        
        yaml_config_specific =  yaml_config_base + 'sweep_' + runID + ".yaml"
        with open(yaml_config_specific, 'w') as file:
            yaml.dump(specific_config, file, default_flow_style=False)