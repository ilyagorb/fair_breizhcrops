import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from .urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, FILESIZES, SHP_URLs, H5_URLs, RAW_CSV_URL
from ..utils import download_file, unzip, untar

BANDS = {
    "L1C": ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id'],
    "L2A": ['doa', 'id', 'code_cultu', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT']
}

SELECTED_BANDS = {
			"L1C": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 
						'QA10', 'QA20', 'QA60', 'doa'],
			"L2A": ['doa','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
					'CLD', 'EDG', 'SAT',]
}

PADDING_VALUE = -1


class BreizhCrops(Dataset):

    def __init__(self,
                 region,
                 root="breizhcrops_dataset",
                 year=2017, level="L1C",
                 csv_file_name=None, 
                 transform=None,
                 target_transform=None,
                 filter_length=0,
                 verbose=False,
                 load_timeseries=True,
                 recompile_h5_from_csv=False,
                 preload_ram=False,
                 include_area=False,
                 use_area_feature=False,
                 exclude_bands=None):
        """
        :param region: dataset region. choose from "frh01", "frh02", "frh03", "frh04", "belle-ile"
        :param root: where the data will be stored. defaults to `./breizhcrops_dataset`
        :param year: year of the data. currently only `2017`
        :param level: Sentinel 2 processing level. Either `L1C` (top of atmosphere) or `L2A` (bottom of atmosphere)
        :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        :param target_transform: a transformation function applied to the label.
        :param filter_length: time series shorter than `filter_length` will be ignored
        :param bool verbose: verbosity flag
        :param bool load_timeseries: if False, no time series data will be loaded. Only index file and class initialization. Used mostly for tests
        :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.
        """
        assert year in [2017, 2018]
        assert level in ["L1C", "L2A"]
        assert region in ["frh01", "frh02", "frh03", "frh04", "belle-ile", "kermaux", "OHIT"]
        
        self.region = region.lower()
        self.bands = BANDS[level]
        self.selected_bands = SELECTED_BANDS[level]

        if exclude_bands:
            self.selected_bands = [band for band in self.selected_bands if band not in exclude_bands]

        if transform is None:
            transform = get_default_transform(level,self.selected_bands)
        if target_transform is None:
            target_transform = get_default_target_transform()
        self.transform = transform
        self.target_transform = target_transform


        

        self.verbose = verbose
        self.year = year
        self.level = level
        
        self.include_area = include_area
        self.use_area_feature = use_area_feature
        self.area_normalized = None

        if verbose:
            print(f"Initializing BreizhCrops region {region}, year {year}, level {level}")

        self.root = root
        self.h5path, self.default_indexfile, self.codesfile, self.shapefile, self.classmapping, self.csvfolder = \
            self.build_folder_structure(self.root, self.year, self.level, self.region)

        self.load_classmapping(self.classmapping)

        if csv_file_name:
            # Check if the file exists at the specified path
            if not os.path.exists(csv_file_name):
                raise FileNotFoundError(f"The specified CSV file {csv_file_name} does not exist.")
            self.indexfile = csv_file_name
        else:
            self.indexfile = self.default_indexfile
            # If the default index file does not exist, download it
            if not os.path.exists(self.indexfile):
                download_file(INDEX_FILE_URLs[year][level][region], self.indexfile)
        
        self.index = pd.read_csv(self.indexfile, index_col=0)
        
        print(f"Expected h5 file path: {self.h5path}")
        
        if os.path.exists(self.h5path):
            h5_database_ok = True
            #h5_database_ok = os.path.getsize(self.h5path) == FILESIZES[year][level][region]
            
        else:
            h5_database_ok = False
            if region == "OHIT":
                raise FileNotFoundError(f"Manually place the OHIT .h5 file at {self.h5path}")
            


        if not os.path.exists(self.indexfile):
            download_file(INDEX_FILE_URLs[year][level][region], self.indexfile)

        if not h5_database_ok and recompile_h5_from_csv and load_timeseries:
            self.download_csv_files()
            self.write_index()
            self.write_h5_database_from_csv(self.index)
        if not h5_database_ok and not recompile_h5_from_csv and load_timeseries:
            self.download_h5_database()

        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        if self.index.index.name != "idx":
            self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")
            
        

        self.update_class_ids()

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            download_file(CODESURL, self.codesfile)
        self.codes = pd.read_csv(self.codesfile, delimiter=";", index_col=0)


        if preload_ram:
            self.X_list = list()
            with h5py.File(self.h5path, "r") as dataset:
                for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
                    self.X_list.append(np.array(dataset[(row.path)]))
        else:
            self.X_list = None

        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

        if "classid" not in self.index.columns or "classname" not in self.index.columns or "region" not in self.index.columns:
            valid_codes = self.index["CODE_CULTU"].isin(self.mapping.index)
            self.index = self.index[valid_codes]

            mapping_result = self.index["CODE_CULTU"].apply(lambda code: pd.Series({
                "classid": self.mapping.loc[code, 'id'],
                "classname": self.mapping.loc[code, 'classname']
            }))

            self.index = pd.concat([self.index, mapping_result], axis=1)
            self.index["region"] = self.region

            
        # Save the updated index
        self.index.to_csv(self.indexfile)

        self.get_codes()

        
    def get_adjusted_class_distribution(self):
        # Assuming self.index or a similar attribute holds the dataframe with adjusted class IDs
        class_distribution = self.index['classid'].value_counts().to_dict()
        return class_distribution


    def update_class_ids(self, global_mapping=None):
        """
        Update class IDs based on a global mapping or unique class IDs in the dataset.
        """
        if global_mapping is not None:
            self.mapping = global_mapping.copy()
        else:

            # Step 1: Identify present CODE_CULTU values in the dataset
            present_codes = self.index['CODE_CULTU'].unique()

            # Step 2: Map present CODE_CULTU to their corresponding class IDs based on the original classmapping
            # This ensures continuity and respects the original order
            ordered_class_ids = self.mapping.loc[present_codes]['id'].sort_values().unique()

            # Step 3: Create a new mapping for present class IDs to ensure continuity and respect original order
            self.new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(ordered_class_ids)}

            # Correctly apply the new mapping to the self.mapping['id'] before attempting to use 'new_id'
            self.mapping['id'] = self.mapping['id'].map(self.new_id_mapping)
            

        # Ensure 'new_id' column is correctly established in self.mapping
        self.mapping['new_id'] = self.mapping['id']  # At this point, 'id' is effectively the new_id

            
        # Step 4: Apply the new mapping to the index
        self.index['classid'] = self.index['CODE_CULTU'].map(self.mapping['id']).astype(int)

        # Step 5: Directly map new_id to unique classname.


        # Exclude entries with NaN new_id before creating the mapping
        unique_classnames_mapping = self.mapping[['new_id', 'classname']].dropna().drop_duplicates().sort_values('new_id').set_index('new_id')

        # Extract a list of unique class names from the valid mapping, excluding those with NaN new_id
        self.classname = unique_classnames_mapping['classname'].tolist()


                # Logging for verification
        print("Updated class IDs and class names to ensure uniqueness and correct order:")
        print(unique_classnames_mapping)

            # Logging for verification
        print("Updated class IDs mapping to maintain original order:\n", self.new_id_mapping)
        print("Updated index preview:\n", self.index.head())
        print("Updated class mapping preview:\n", self.mapping)

            # Update self.classes and self.nclasses
        self.classes = np.unique(self.index['classid'])
        self.nclasses = len(self.classes)
        print(f"Updated number of classes to {self.nclasses}")



    def download_csv_files(self):
        zipped_file = os.path.join(self.root, str(self.year), self.level, f"{self.region}.zip")
        download_file(RAW_CSV_URL[self.year][self.level][self.region], zipped_file)
        unzip(zipped_file, self.csvfolder)
        os.remove(zipped_file)

    def build_folder_structure(self, root, year, level, region):
        """
        folder structure

        <root>
           codes.csv
           classmapping.csv
           <year>
              <region>.shp
              <level>
                 <region>.csv
                 <region>.h5
                 <region>
                     <csv>
                         123123.csv
                         123125.csv
                         ...
        """
        year = str(year)

        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        h5path = os.path.join(root, year, level, f"{region}.h5")
        indexfile = os.path.join(root, year, level, region + ".csv")
        codesfile = os.path.join(root, "codes.csv")
        shapefile = os.path.join(root, year, f"{region}.shp")
        classmapping = os.path.join(root, "classmapping.csv")
        csvfolder = os.path.join(root, year, level, region, "csv")

        return h5path, indexfile, codesfile, shapefile, classmapping, csvfolder

    def get_fid(self, idx):
        return self.index[self.index["idx"] == idx].index[0]

    def download_h5_database(self):

        print(f"downloading {self.h5path}.tar.gz")
        download_file(H5_URLs[self.year][self.level][self.region], self.h5path + ".tar.gz", overwrite=True)
        print(f"extracting {self.h5path}.tar.gz to {self.h5path}")
        untar(self.h5path + ".tar.gz")
        print(f"removing {self.h5path}.tar.gz")
        os.remove(self.h5path + ".tar.gz")
        print(f"checking integrity by file size...")
        assert os.path.getsize(self.h5path) == FILESIZES[self.year][self.level][self.region]
        print("ok!")

    def write_h5_database_from_csv(self, index):
        with h5py.File(self.h5path, "w") as dataset:
            for idx, row in tqdm(index.iterrows(), total=len(index), desc=f"writing {self.h5path}"):                             
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def download_geodataframe(self):
        targzfile = os.path.join(os.path.dirname(self.shapefile), self.region + ".tar.gz")
        download_file(SHP_URLs[self.year][self.region], targzfile)
        untar(targzfile)
        os.remove(targzfile)

    def geodataframe(self):

        if not os.path.exists(self.shapefile):
            self.download_geodataframe()

        geodataframe = gpd.GeoDataFrame(self.index.set_index("id"))

        gdf = gpd.read_file(self.shapefile)

        # 2018 shapefile calls ID ID_PARCEL: rename if necessary
        gdf = gdf.rename(columns={"ID_PARCEL": "ID"})

        # copy geometry from shapefile to index file
        geom = gdf.set_index("ID")
        geom.index.name = "id"
        geodataframe["geometry"] = geom["geometry"]
        geodataframe.crs = geom.crs

        return geodataframe.reset_index()

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.verbose:
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def load_raw(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
               'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""
        sample = pd.read_csv(os.path.join(self.csvfolder, os.path.basename(csv_file)), index_col=0).dropna()

        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype(int)
        sample = sample.groupby(by="doa").first().reset_index()

        return sample

    def load(self, csv_file):
        sample = self.load_raw(csv_file)
        
        selected_bands = self.selected_bands
        X = np.array(sample[selected_bands].values)	
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0
            X = X[~t_without_nans]

        return X

    def load_culturecode_and_id(self, csv_file):
        sample = self.load_raw(csv_file)
        X = np.array(sample.values)
		
        if self.level=="L1C":
            cc_index = self.bands.index("label")
        else:
            cc_index = self.bands.index("code_cultu")
        id_index = self.bands.index("id")

        if len(X) > 0:
            field_id = X[0, id_index]
            culture_code = X[0, cc_index]
            return culture_code, field_id

        else:
            return None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]

        if self.X_list is None:
            with h5py.File(self.h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id
        
                
        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)
            


        if self.include_area:
            # Check for 'area' attribute and if row.id is in its index
            if not hasattr(self, 'area') or row.id not in self.area.index:
                raise ValueError(f"Area information missing for row ID {row.id}")
            sample_area = self.area.loc[row.id]
        
        if self.use_area_feature:
            # Ensure the normalized area data is available
            if self.area_normalized is None or row.id not in self.area_normalized.index:
                raise ValueError(f"Normalized area information missing for row ID {row.id}")
            # Fetch the normalized area value
            normalized_area = self.area_normalized.loc[row.id]
            # Append the normalized area to the feature vector
            # Create a column vector with the same value (normalized_area) repeated for each time step

            normalized_area_column = torch.full((X.shape[0], 1), normalized_area, dtype=torch.float32)
            X = torch.cat((X, normalized_area_column), dim=1)


            
        if self.include_area:
            return X, y, sample_area, row.id
        
        else:
            return X, y, row.id  # Return row.id without area


    def write_index(self):
        csv_files = os.listdir(self.csvfolder)
        listcsv_statistics = list()
        i = 1

        for csv_file in tqdm(csv_files):
            if self.level == "L1C":
                cld_index = SELECTED_BANDS["L1C"].index("QA60")
            elif self.level == "L2A":
                cld_index = SELECTED_BANDS["L2A"].index("CLD")

            X = self.load(os.path.join(self.csvfolder, csv_file))
            culturecode, id = self.load_culturecode_and_id(os.path.join(self.csvfolder, csv_file))

            if culturecode is None or id is None:
                continue

            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),
                    id=id,
                    CODE_CULTU=culturecode,
                    path=os.path.join(self.csvfolder, f"{id}" + ".csv"),
                    idx=i,
                    sequencelength=len(X)
                )
            )
            i += 1

        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)
        
    def load_and_merge_geojson_attributes(self, geojson_path, attribute_names):
        """
        Load specified attributes from a GeoJSON file and merge them into the dataset.

        Parameters:
        - geojson_path: Path to the GeoJSON file.
        - attribute_names: List of attribute names (columns) to load from the GeoJSON. 
                           The 'id' column is mandatory for merging.
        """
        # Load the GeoJSON file
        additional_info_gdf = gpd.read_file(geojson_path)
        
        # Ensure 'id' is included for merging
        if 'id' not in attribute_names:
            attribute_names.append('id')
        
        # Filter the GeoDataFrame for the specified attributes
        additional_info_gdf = additional_info_gdf[attribute_names]
        
        # Merge the attributes into the index DataFrame
        self.index = pd.merge(self.index, additional_info_gdf, on='id', how='left')
        
        # Update the list of extra attribute names, excluding 'id'
        self.extra_attribute_names = [name for name in attribute_names if name != 'id']



def get_default_transform(level, selected_bands):

    #padded_value = PADDING_VALUE
    sequencelength = 45

    # Filter the spectral bands from the selected bands based on the level
    if level == "L1C":
        spectral_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    elif level == "L2A":
        spectral_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    # Only keep the selected bands that are in the spectral bands list
    filtered_bands = [band for band in selected_bands if band in spectral_bands]

    # Get the indices of the filtered bands
    selected_band_idxs = np.array([selected_bands.index(b) for b in filtered_bands])

    def transform(x):
        #x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor)
    return transform



def get_default_target_transform():
    return lambda y: torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    BreizhCrops(region="frh03", root="/tmp", load_timeseries=False, level="L2A",recompile_h5_from_csv=True, year=2018)
