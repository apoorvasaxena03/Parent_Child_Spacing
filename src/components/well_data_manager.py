# %%
# Importing pandas package for data manipulation and analysis
import pandas as pd
pd.set_option('display.max_columns', None) # Set the maximum number of columns to display to None

import numpy as np # Importing numpy package for numerical operations

from typing import Dict, Union, Optional # Importing specific types from typing module

import time # Importing Time Module

import pyproj # Importing pyproj package

from src.custom_logger import CustomLogger # Importing CustomLogger class from custom

import os # Importing os module for operating system dependent functionality

# Importing necessary modules for plotting and data manipulation
import matplotlib.pyplot as plt # Importing matplotlib.pyplot for plotting

# Setting matplotlib to inline mode for Jupyter notebooks
#%matplotlib inline

#%config InlineBackend.figure_format = 'svg' # Configuring inline backend to use SVG format for figures

# %%
class WellDataLoader:
    """
    A class to load well header and directional survey data either from file (CSV/Excel)
    or directly from a database. It supports flexible column mapping for file-based inputs.
    """

    def __init__(
        self,
        db: Optional[object] = None,
        log_dir: str = "./logs"
    ):
        self.db = db
        self.logger = CustomLogger("well_data_loader", "WellDataLoaderLogger", log_dir).get_logger()
        self.header_df = pd.DataFrame()

    def load_data_from_file(
        self,
        file_path: str,
        required_columns: Dict[str, str],
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        try:
            usecols = list(required_columns.values())
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, dtype=dtype, usecols=usecols)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, dtype=dtype, usecols=usecols)
            else:
                raise ValueError("Unsupported file type. Use CSV or Excel.")

            missing = [val for val in required_columns.values() if val not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in file: {missing}")

            return df.rename(columns={v: k for k, v in required_columns.items()})

        except Exception as e:
            self.logger.error(f"Failed to load data from file {file_path}: {e}")
            raise

    def get_header_data(
        self,
        source: Optional[Union[str, pd.DataFrame]] = None,
        column_map: Optional[Dict[str, str]] = None,
        basin: str = "MB",
        start_year: int = 2019,
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            self.logger.info("Using provided header DataFrame.")
            df = source
        elif isinstance(source, str) and os.path.exists(source):
            if not column_map:
                raise ValueError("Column map must be provided when reading from file.")
            self.logger.info(f"Loading header data from file: {source}")
            df = self.load_data_from_file(source, column_map, dtype=dtype)
        elif source is None:
            self.logger.info("Loading header data from SQL.")
            df = self._query_header_from_db(basin, start_year)
        else:
            raise ValueError("Invalid input: provide either a file path, DataFrame, or SQL query.")

        self.header_df = df
        return df

    def get_directional_data(
        self,
        source: Optional[str] = None,
        column_map: Optional[Dict[str, str]] = None,
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        if source and os.path.exists(source):
            if not column_map:
                raise ValueError("Column map must be provided when reading from file.")
            self.logger.info(f"Loading directional data from file: {source}")
            return self.load_data_from_file(source, column_map, dtype=dtype)
        elif source is None:
            self.logger.info("Loading directional data from SQL.")
            return self._query_directional_from_db()
        else:
            raise ValueError("Provide either a file path or SQL query for directional data.")

    def _query_header_from_db(self, basin: str, start_year: int) -> pd.DataFrame:
        query = f"""
        SELECT
            api14 AS uwi, 
            leaseName AS lease_name,
            wellName AS well_name,
            wellNumber AS well_num,
            currentOperator AS operator,
            customString2 AS rsv_cat,
            customString0 AS bench,
            DATE(firstProdDate) AS first_prod_date,
            holeDirection AS hole_direction,
            surfaceLatitude AS surface_lat,
            surfaceLongitude AS surface_lon
        FROM Combocurve.export.wells
        WHERE basin = '{basin}'
          AND customString2 in ("01PDP", "02PDNP", "02PA") 
          AND holeDirection = 'H' 
          AND YEAR(DATE(firstProdDate)) >= {start_year}
        """
        try:
            self.db.connect()
            df = self.db.execute_query(query)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving header data from databricks: {e}")
            raise
        finally:
            self.db.close_connection()

    def _query_directional_from_db(self) -> pd.DataFrame:
        if self.header_df.empty or 'uwi' not in self.header_df.columns:
            raise ValueError("Header data must be loaded before querying directional data, and must contain a 'uwi' column.")

        uwis = ", ".join(f"'{id}'" for id in self.header_df['uwi'].unique())
        query = f"""
        SELECT
            uwi, 
            station_md_uscust AS md, 
            station_tvd_uscust AS tvd,
            inclination, 
            azimuth, 
            latitude, 
            longitude, 
            x_offset_uscust AS `deviation_E/W`,
            ew_direction as `E/W`,
            y_offset_uscust AS `deviation_N/S`,
            ns_direction  as `N/S`,
            point_type as point_type_name
        FROM ihs_sp.well.well_directional_survey_station
        WHERE uwi IN ({uwis})
        ORDER BY uwi, md;
        """
        try:
            self.db.connect()
            df = self.db.execute_query(query)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving directional data from databricks: {e}")
            raise
        finally:
            self.db.close_connection()


# %%
class GeoSurveyProcessor:
    """
    A class for processing directional survey data and performing geospatial transformations
    such as converting lat/lon to UTM, filtering heel points, and extracting key well locations.
    """
    def __init__(self, log_dir: str = "./logs",):
        """
        Initializes the GeoSurveyProcessor with optional header data and log directory.
        :param log_dir: Directory for logging.
        """
        self.logger = CustomLogger("geo_processor", "GeoLogger", log_dir).get_logger()  # Custom logger
        self.logger.info("GeoSurveyProcessor initialized.")

    def determine_utm_zone(self, longitude: float) -> int:
        """
        Determines the UTM zone based on a given longitude.
        """
        return int((longitude + 180) / 6) + 1
        
    def convert_utm_to_latlon(self, 
                              df: pd.DataFrame, x_col: str = "x", y_col: str = "y", 
                              zone_col: str = "utm_zone", epsg_col: str = "epsg_code", 
                              lat_col: str = "latitude", lon_col: str = "longitude",
                              round_output: bool = True) -> pd.DataFrame:
        """
        Converts UTM (x, y) coordinates back to lat/lon using EPSG codes or UTM zones.

        Parameters:
        - df: DataFrame with UTM x/y in feet and a zone identifier.
        - x_col, y_col: Column names for UTM coordinates (in feet).
        - zone_col: Column containing UTM zone numbers.
        - epsg_col: Optional EPSG code column (e.g., 'EPSG:32613'). If not present, it will be constructed from zone_col.
        - round_output: Whether to round output lat/lon to 8 decimal places.

        Returns:
        - DataFrame with added columns: 'lon_from_utm' and 'lat_from_utm'
        """

        df = df.copy()
        
        if epsg_col not in df.columns:
            df[epsg_col] = df[zone_col].apply(lambda z: f"EPSG:326{int(z)}")

        df[lat_col] = np.nan
        df[lon_col] = np.nan

        for epsg in df[epsg_col].unique():
            mask = df[epsg_col] == epsg

            # Convert feet to meters
            x_m = df.loc[mask, x_col] / 3.28084
            y_m = df.loc[mask, y_col] / 3.28084

            transformer = pyproj.Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_m.values, y_m.values)

            if round_output:
                lon = np.round(lon, 8)
                lat = np.round(lat, 8)

            df.loc[mask, lat_col] = lat
            df.loc[mask, lon_col] = lon

        self.logger.info(f"✅ Back-converted UTM to lat/lon for {len(df)} rows.")
        return df
    
    def compute_utm_coordinates(self, df: pd.DataFrame, surface_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes UTM (x, y, z) coordinates in feet. Uses per-row lat/lon if available.
        Otherwise uses surface_df to compute UTM using deviation displacements.

        Adds EPSG code used per row. Optionally back-computes lat/lon from UTM for verification.

        Parameters:
        - df: pd.DataFrame with directional survey data.
        - surface_df: Optional pd.DataFrame with ['uwi', 'surface_lat', 'surface_lon'].

        Returns:
        - pd.DataFrame with UTM coordinates 'x', 'y', 'z', 'utm_zone', 'epsg_code', and optionally back-computed lat/lon.
        """
        
        start_time = time.time()
        df = df.sort_values(by=["uwi", "md"]).copy()

        df["x"], df["y"] = np.zeros(len(df)), np.zeros(len(df))

        if "latitude" in df.columns and "longitude" in df.columns:
            self.logger.info("✅ Using lat/lon from input DataFrame.")
            df["utm_zone"] = df["longitude"].apply(self.determine_utm_zone)

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "longitude"].values
                lat = df.loc[mask, "latitude"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                df.loc[mask, "x"] = easting_m * 3.28084
                df.loc[mask, "y"] = northing_m * 3.28084
                df.loc[mask, "epsg_code"] = epsg_code

        elif surface_df is not None:
            self.logger.info("🧭 Lat/Lon not available — using surface_df and displacements.")
            
            required_cols = {"uwi", "surface_lat", "surface_lon"}
            
            if not required_cols.issubset(surface_df.columns):
                raise ValueError(f"surface_df must contain {required_cols}")

            df = df.merge(surface_df, on="uwi", how="left")
            df["utm_zone"] = df["surface_lon"].apply(self.determine_utm_zone)

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "surface_lon"].values
                lat = df.loc[mask, "surface_lat"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                # Convert to feet
                easting_ft = easting_m * 3.28084
                northing_ft = northing_m * 3.28084

                ew_sign = df.loc[mask, "E/W"].map({"E": 1, "W": -1}).fillna(0)
                ns_sign = df.loc[mask, "N/S"].map({"N": 1, "S": -1}).fillna(0)

                df.loc[mask, "x"] = easting_ft + df.loc[mask, "deviation_E/W"] * ew_sign
                df.loc[mask, "y"] = northing_ft + df.loc[mask, "deviation_N/S"] * ns_sign
                df.loc[mask, "epsg_code"] = epsg_code

            # Back-convert to lat/lon using x/y and epsg_code
            df = self.convert_utm_to_latlon(df)

        else:
            raise ValueError("Either lat/lon must be present in df, or surface_df must be provided.")

        df["z"] = -df["tvd"]

        self.logger.info(f"✅ UTM coordinate computation complete in {time.time() - start_time:.2f} sec.")
        return df
    
    def filter_after_heel_point(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataframe to include all rows for each uwi where the first occurrence 
        of either '80' or 'heel' appears in the point_type column and all subsequent rows.

        Parameters:
        df (pd.DataFrame): A dataframe containing directional survey data with a 'uwi' column and 'point_type' column.

        Returns:
        pd.DataFrame: Filtered dataframe containing rows from the first occurrence of '80' or 'heel' onward.
        """
        # Ensure the data is sorted by MD in ascending order
        df = df.sort_values(by=["uwi", "md"], ascending=True).copy()

        # Convert 'point_type_name' to lowercase and check for '80' or 'heel'
        mask = df['point_type_name'].str.lower().str.contains(r'80|heel', regex=True, na=False)

        # Identify the first occurrence for each uwi
        idx_start = df[mask].groupby('uwi', sort=False).head(1).index

        # Create a mapping of uwi to the starting index
        start_idx_map = dict(zip(df.loc[idx_start, 'uwi'], idx_start))

        # Create a boolean mask using NumPy to filter rows
        uwis = df['uwi'].values
        indices = np.arange(len(df))

        # Get the minimum start index for each row's uwi
        start_indices = np.vectorize(start_idx_map.get, otypes=[float])(uwis)

        # Mask rows where index is greater than or equal to the start index
        valid_rows = indices >= start_indices

        return df[valid_rows].reset_index(drop=True)
    
    def get_heel_toe_midpoints_latlon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the heel, toe, and mid-point latitude/longitude for each uwi in the well trajectory DataFrame
        that has been filtered to have lateral section of the well.

        Parameters:
        df: pd.DataFrame
            DataFrame containing well trajectory data, including 'uwi', 'md', 'latitude', and 'longitude'.

        Returns:
        pd.DataFrame
            A DataFrame with 'uwi', 'Heel_Lat', 'Heel_Lon', 'Toe_Lat', 'Toe_Lon', 'Mid_Lat', 'Mid_Lon'.

        Example:
        >>> data = {
        ...     "uwi": [1001, 1001, 1001, 1002, 1002],
        ...     "md": [5000, 5100, 5200, 6000, 6100],
        ...     "latitude": [31.388, 31.389, 31.387, 31.400, 31.401],
        ...     "longitude": [-103.314, -103.315, -103.316, -103.318, -103.319]
        ... }
        >>> df = pd.DataFrame(data)
        >>> extract_heel_toe_mid_lat_lon(df)
        uwi  Heel_Lat  Heel_Lon  Toe_Lat  Toe_Lon  Mid_Lat  Mid_Lon
        0     1001    31.388  -103.314   31.387  -103.316  31.3875 -103.315
        1     1002    31.400  -103.318   31.401  -103.319  31.4005 -103.3185
        """
        # Getting DataFrame with only the rows after the heel point
        df = self.filter_after_heel_point(df)

        # Group by 'uwi' and extract heel/toe lat/lon
        heel_toe_df = (
            df.groupby("uwi")
            .agg(
                heel_lat=("latitude", "first"),
                heel_lon=("longitude", "first"),
                toe_lat=("latitude", "last"),
                toe_lon=("longitude", "last"),
            )
            .reset_index()
        )

        # Calculate midpoints
        heel_toe_df["mid_Lat"] = (heel_toe_df["heel_lat"] + heel_toe_df["toe_lat"]) / 2
        heel_toe_df["mid_Lon"] = (heel_toe_df["heel_lon"] + heel_toe_df["toe_lon"]) / 2

        return heel_toe_df
    
    def plot_utm_trajectory(
        self,
        df: pd.DataFrame,
        plot_3d: bool = True,
        uwis: Optional[Union[list, str]] = None
    ) -> None:
        """
        Visualizes the UTM trajectory (2D or 3D) for one or multiple wells.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'x', 'y', 'z', and 'uwi' columns.
        - plot_3d (bool): Whether to plot in 3D (True) or 2D (False). Defaults to True.
        - uwis (Optional[list or str]): One or more specific uwis to filter and plot. Defaults to all wells.
        """
        if uwis is not None:
            if isinstance(uwis, str):
                uwis = [uwis]
            df = df[df["uwi"].isin(uwis)]

        fig = plt.figure(figsize=(10, 10))
        title = "3D Well Trajectory (UTM ft)" if plot_3d else "2D Well Plan View (x-y, UTM ft)"
        fig.suptitle(title, fontsize=14)

        if plot_3d:
            ax = fig.add_subplot(111, projection='3d')
            for uwi, group in df.groupby("uwi"):
                ax.plot(group["x"], group["y"], group["z"], label=str(uwi))
            ax.set_zlabel("Z (ft, -TVD)")
        else:
            ax = fig.add_subplot(111)
            for uwi, group in df.groupby("uwi"):
                ax.plot(group["x"], group["y"], label=str(uwi))
        
        ax.set_xlabel("X (Easting, ft)")
        ax.set_ylabel("Y (Northing, ft)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


