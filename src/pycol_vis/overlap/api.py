from .overlap_measures import *


class OverlapAPI:

    def __init__(self, parent):
        self.parent = parent
        self.embeddings_api = parent.embeddings

    def handle_embs_reduction(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        embs = self.embeddings_api.embed_images(emb_type=emb_type, layer_index=layer_index, num_workers=num_workers, device=device)

        if(embs is None):
            return None

        if(reduction_type is not None):
            embs = self.embeddings_api.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_components=n_components)

        return embs

    def put_overlap_measures_in_dic(self, measure_name, measure_value, emb_type='efficient_net', layer_index=-1):
        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic[measure_name + '_' + emb_type + '_layer' + layer_index_str] = measure_value


    def all_overlap_measures(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_samples=50, n_components=10, num_workers=0, device=None):
        '''
        Calculate all overlap measures for the specified embedding type and layer index, and store the results in the self.overlap_measures_dic dictionary.
        '''
        

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        m_var = m_var_measure(embeddings=embs,labels=self.parent.images['class'].values)
        m_sep = m_sep_measure(embeddings=embs,labels=self.parent.images['class'].values)
        tabular_n2 = tabular_measure(embeddings=embs, labels=self.parent.images['class'].values, measure='n2')
        tabular_kdn = tabular_measure(embeddings=embs, labels=self.parent.images['class'].values, measure='kdn')
        tabular_lsc = tabular_measure(embeddings=embs, labels=self.parent.images['class'].values, measure='lsc')
        auls = auls_measure(embeddings=embs, labels=self.parent.images['class'].values, n_samples=n_samples)
        csg = csg_measure(embeddings=embs, labels=self.parent.images['class'].values, n_samples=n_samples, auls=False)
        cms_auls = csg_measure(embeddings=embs, labels=self.parent.images['class'].values, n_samples=n_samples, auls=True)

        self.put_overlap_measures_in_dic('m_var', m_var, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('m_sep', m_sep, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('tabular_n2', tabular_n2, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('tabular_kdn', tabular_kdn, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('tabular_lsc', tabular_lsc, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('auls', auls, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('csg', csg, emb_type=emb_type, layer_index=layer_index)
        self.put_overlap_measures_in_dic('cms_auls', cms_auls, emb_type=emb_type, layer_index=layer_index)
        
        #self.entropy_based_measure(emb_type=emb_type, layer_index=layer_index,reduction_type=reduction_type,reduction_method=reduction_method,n_components=n_components,num_workers=num_workers,device=device)
        #self.entropic_graphs_measure(emb_type=emb_type,k=5, layer_index=layer_index,reduction_type=reduction_type,reduction_method=reduction_method,n_components=n_components,num_workers=num_workers,device=device)

    def m_var_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Compute the M_var measure of class variability in the embedding space.

        M_var is calculated using the normalized within-class scatter matrix (S_w_hat) in the embedding space, which captures the variability of samples within each class. 
        A lower M_var value indicates that samples within the same class are more tightly clustered together, suggesting better class separability.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_var. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 10.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated M_var value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")


        measure = m_var_measure(embeddings=embs,labels=self.parent.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"

        self.parent.overlap_measures_dic['m_var_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def m_sep_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Compute the M_sep measure of class separability in the embedding space.

        M_sep is calculated using the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat) in the embedding space.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_sep. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 10.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        Returns:
        - float: The calculated M_sep value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = m_sep_measure(embeddings=embs,labels=self.parent.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['m_sep_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def tabular_measure(self, layer_index=-1, reduction_type='pca', reduction_method=None, emb_type='efficient_net', measure='kdn', n_samples=None, n_components=10, num_workers=0, device=None):
        '''
        Calculate overlap measures using the pycol complexity libray.

        A lower value of the meaure indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.

        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the overlap measures. Options are 'pca', 'tsne', or 'custom'. 
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - emb_type (str): The type of embeddings to use for the calculation. 
        - measure (str): The specific overlap measure to calculate. Options are 'n2', 'kdn', or 'lsc'. Each measure captures different aspects of class overlap and complexity in the feature space.
        - n_samples (int): The number of samples to use for calculating the complexity measure.
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 2.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = tabular_measure(embeddings=embs, labels=self.parent.images['class'].values, measure=measure, n_samples=n_samples)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['tabular_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def auls_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Calculate the AULS complexity measure based on the spectrum of the graph. AULS is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
        A lower AULS value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated AULS complexity score for the dataset based on the specified embedding
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = auls_measure(embeddings=embs, labels=self.parent.images['class'].values, n_samples=n_samples)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['auls_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def csg_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, auls=False, num_workers=0, device=None):
        '''
         Calculate the CSG complexity measure based on the spectrum of the graph. CSG is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
        A lower CSG value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.
        - auls (bool): Whether to calculate the cmsAULS complexity measure instead of CSG. 
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated CSG or csmAULS complexity score for the dataset based on the specified embedding
        '''


        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = csg_measure(embeddings=embs, labels=self.parent.images['class'].values, n_samples=n_samples, auls=auls)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['csg_' + emb_type + '_layer' + layer_index_str] = measure

        return measure
    
    def entropy_based_measure(self, layer_index=-1, reduction_type='pca', reduction_method=None, emb_type='efficient_net', n_components=10, num_workers=0, device=None):

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = entropy_based_measure(embeddings=embs, labels=self.parent.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['entropy_' + emb_type + '_layer' + layer_index_str] = measure

        return measure
    
    def entropic_graphs_measure(self, layer_index=-1, reduction_type='pca', reduction_method=None, emb_type='efficient_net', n_components=10, k=5, num_workers=0, device=None):

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = entropic_graphs_measure(embeddings=embs, labels=self.parent.images['class'].values, k=k)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.parent.overlap_measures_dic['entropic_graphs_' + emb_type + '_layer' + layer_index_str] = measure

        return measure
    
    def fisher_discriminative_map(self):
        measure = fisher_discriminative_map(self.parent.images['image_path'].values,self.parent.images['class'].values)
        return measure
    
    def patch_discriminative_maps(self, image_size=(64, 64), patch_size=8, patch_descriptor_method="all",measure_method = "fisher",aggregation=None, measure_params=None):
        maps = patch_discriminative_maps(self.parent.images['image_path'].values,self.parent.images['class'].values, image_size=image_size, patch_size=patch_size, descriptor_method = patch_descriptor_method, measure_method=measure_method, aggregation=aggregation, measure_params=measure_params)
        return maps
    
    def global_discriminative_descriptors(
        self,
        image_size=(256, 256),
        descriptor_method="all",
        measure_method = "fisher"
    ):
        

        measure = global_discriminative_descriptors(self.parent.images['image_path'].values,
                                                   self.parent.images['class'].values,
                                                   image_size=image_size,
                                                   descriptor_method=descriptor_method,
                                                   measure_method=measure_method
                                                   )
        return measure