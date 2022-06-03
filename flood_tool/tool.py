"""Example module in template package."""
import os

import numpy as np
import pandas as pd
import joblib
from . import geo
from .models import train_flood_model, train_price_model


__all__ = ['Tool']


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels='',
                 household_file=''):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        sample_labels : str, optional
            Filename of a .csv file containing sample data on property
            values and flood risk labels.

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.
        """
        self.new_samples = not sample_labels == ''

        if postcode_file == '':
            postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv'))

        if sample_labels == '':
            sample_labels = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sampled.csv'))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))

        self.postcode_file = postcode_file
        self.sample_labels = sample_labels
        self.household_file = household_file

        self.postcode_df = pd.read_csv(self.postcode_file)
        self.sample_df = pd.read_csv(self.sample_labels)
        self.household_df = pd.read_csv(self.household_file)

    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.

        Example
        -------
        >>> get_easting_northing(['DE2 3DA', 'LN5 7RW'])
                   easting  northing
        postcode
        DE2 3DA   434011.0  330722.0
        LN5 7RW   497441.0  370798.0

        >>> get_easting_northing(['XX1 2XX'])
                  easting  northing
        postcode
        XX2 3XAA      NaN       NaN
        """
        postcode_df = self.postcode_df
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df = postcode_df.set_index('postcode')
        postcode_df = postcode_df[['easting', 'northing']]
        # postcode_df.loc[postcodes]
        # locate the postcode related data
        result_df = postcode_df.loc[postcode_df.index.isin(postcodes)]
        # remaining invalid postcode
        remaining_postcodes = set(postcodes) - set(result_df.index)
        remaining_df = pd.DataFrame({'postcode': list(remaining_postcodes),
                                     'easting': np.full(len(remaining_postcodes), np.nan),
                                     'northing': np.full(len(remaining_postcodes), np.nan)}).set_index('postcode')
        # combine df
        result_df = pd.concat([result_df, remaining_df])
        result_df = result_df.reindex(postcodes)  # reindex by sectors sequence
        return result_df

    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.

        Example
        -------
        >>> get_lat_long(['DE2 3DA', 'LN5 7RW'])
                   latitude  logitude
        postcode
        DE2 3DA   52.872902 -1.496148
        LN5 7RW   53.225300 -0.541897

        >>> get_lat_long(['DE2 3DA', 'LN5 7RW'])
                  latitude  longitude
        postcode
        XX2 3XAA       NaN      NaN
        """
        postcode_df = self.postcode_df
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df = postcode_df.set_index('postcode')
        postcode_df = postcode_df[['easting', 'northing']]
        # get postcode df if valid
        result_df = postcode_df.loc[postcode_df.index.isin(postcodes)]
        result_df = result_df.copy()
        # convert OSGB36 to WGS84
        for index, row in result_df.iterrows():
            lat, log = geo.get_gps_lat_long_from_easting_northing([row[0]], [
                row[1]])
            result_df.loc[index, 'easting'] = lat
            result_df.loc[index, 'northing'] = log
        result_df = result_df.rename(
            columns={'easting': 'latitude', 'northing': 'longitude'})
        # get remaining invalid postcode
        remaining_postcodes = set(postcodes) - set(result_df.index)
        remaining_df = pd.DataFrame({'postcode': list(remaining_postcodes), 'latitude': np.full(len(
            remaining_postcodes), np.nan), 'longitude': np.full(len(remaining_postcodes), np.nan)}).set_index('postcode')
        # combine df
        result_df = pd.concat([result_df, remaining_df])
        result_df = result_df.reindex(postcodes)  # reindex by sectors sequence
        return result_df

    def get_postcode_from_sector(self, sectors):
        """Get a frame of postcodes from a collection
        of input sectors.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of sectors.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input sectors. Invalid sectors (i.e. not in the
            input unlabelled postcodes file) return as NaN.
        """
        postcode_df = self.postcode_df
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df = postcode_df.set_index('sector')
        postcode_df = postcode_df[['postcode']]

        # get sectors df if valid
        result_df = postcode_df.loc[postcode_df.index.isin(sectors)]
        # get remaining invalid sectors
        remaining_sec = set(sectors) - set(result_df.index)
        remaining_df = pd.DataFrame(
            {'sector': list(remaining_sec), 'postcode': np.full(len(remaining_sec), np.nan)}).set_index("sector")
        result_df = pd.concat([result_df, remaining_df])
        result_df = result_df.reindex(sectors)  # reindex by sectors sequence
        return result_df

    @ staticmethod
    def get_flood_class_methods():
        """
        Get a dictionary of available flood probablity classification methods.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_probability method.
        """
        return {
            "Decision Tree": "dt",
            "KNN": "knn",
            "Random Forest": "rdmf",
            "AdaBoost": "ada"
        }

    @ staticmethod
    def get_flood_class_models():
        """
        Returns
        -------
        Dict
            method to model
        """
        models = {}
        names = [("knn", "knn_model.sav"), ("dt", "decision_tree_model.sav"),
                 ("rdmf", "random_forest_model.sav"), ("ada", "ada_model.sav")]
        for (k, filename) in names:
            models[k] = os.sep.join((os.path.dirname(__file__),
                                     'resources',
                                    'models',
                                     'flood_risk',
                                     filename))
        return models

    def get_flood_class(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_probability_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        methods = ["dt", "knn", "rdmf", "ada"]

        # get method name
        if type(method) == int:
            if method >= len(methods) or method < 0:
                raise NotImplementedError
            method = methods[method]

        if method not in methods:
            raise NotImplementedError  # unknown method

        if self.new_samples:  # re-train data if use new samples
            data = train_flood_model.preprocess_data(self.sample_df)
            loaded_model = train_flood_model.get_model(method, data)
        else:  # use default trained model
            loaded_model = joblib.load(self.get_flood_class_models()[method])
        east_north = self.get_easting_northing(postcodes)
        pred = loaded_model.predict(east_north)
        return pd.Series(data=pred,
                         index=np.asarray(postcodes),
                         name='riskLabel')

    @ staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """
        return {
            'Linear Regression': 'lr',
            'Decision Tree Regressor': 'dt',
            'Random Forest Regressor': 'rfr',
            'SV Regressor': 'sv'
        }

    @ staticmethod
    def get_house_price_models():
        """
        Returns
        -------
        Dict
            method to model
        """
        models = {}
        names = [("lr", "linear_regression.sav"), ("dt", "decision_tree_regressor.sav"),
                 ("rfr", "random_forest_regressor.sav"), ("sv", "svr.sav")]
        for (k, filename) in names:
            models[k] = os.sep.join((os.path.dirname(__file__),
                                     'resources',
                                    'models',
                                     'median_price',
                                     filename))
        return models

    def get_median_house_price_estimate(self, postcodes, method=0):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """
        methods = ["lr", "dt", "rfr", "sv"]

        # get method name based on int
        if type(method) == int:
            if method >= len(methods) or method < 0:
                raise NotImplementedError
            method = methods[method]

        if method not in methods:
            raise NotImplementedError  # unknown method

        if self.new_samples:  # re-train data if use new samples
            data = train_flood_model.preprocess_data(self.sample_df)
            loaded_model = train_price_model.get_model(method, data)
        else:  # use default trained model
            loaded_model = joblib.load(self.get_house_price_models()[method])
        east_north = self.get_easting_northing(postcodes)
        try:
            pred = loaded_model.predict(east_north)
            pred = pred * pred  # inverse transform np.sqrt
        except ValueError:
            # ValueError occurs when the postcode is invalid (i.e. not in the input unlabelled postcode file)
            return pd.Series([np.nan]).set_axis(postcodes)
        return pd.Series(data=pred,
                         index=np.asarray(postcodes),
                         name="medianPrice"
                         )

    def get_total_value(self, locations):
        """
        Return a series of estimates of the total property values
        of a collection of postcode units or sectors.


        Parameters
        ----------

        locations : sequence of strs
            Sequence of postcode units or sectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        codes = locations[0].split(' ')
        len_secondpart = len(codes[1])
        if len_secondpart == 1:
            # Sequence of sectors
            postcodes = self.get_postcode_from_sector(locations)
            total_value_serires = self.get_median_house_price_estimate(
                postcodes.postcode)
            total_value_serires.index = locations
        else:
            # Sequence of postcode units
            total_value_serires = self.get_median_house_price_estimate(
                locations)
        return total_value_serires

    def get_annual_flood_risk(self, postcodes,  risk_labels=None):
        """
        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.


        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood probability classifiers, as
            predicted by get_flood_probability.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """
        risk_dict = {1: 0.0001, 2: 0.0005, 3: 0.001, 4: 0.005,
                     5: 0.01, 6: 0.015, 7: 0.02, 8: 0.03, 9: 0.04, 10: 0.05}
        risk_classes = self.get_flood_class(postcodes)
        total_values = self.get_total_value(postcodes)
        risk_values = []
        for i, j in zip(risk_classes, total_values):
            risk_values.append(0.05*j*risk_dict[i])
        return pd.Series(risk_values, index=postcodes, name="annualFloodRisk")

    def get_combined_data(self):
        """Get lat, log, flood_prob and annual_flood_risk from unlabelled
        postcode file

        Returns
        -------
        pandas.DataFrame

        Example
        -------
        >>> get_combined_data()
                 latitude  longitude  riskLabel  annualFloodRisk  floodProb
        YO62 4LS  54.146810  -0.966109          1         1.180043     0.00001
        DE2 3DA   52.872902  -1.496148          1         1.586082     0.00001
        """
        # Obtain postcode_df with flood_prob, annual_flood_risk
        postcode_df = self.get_lat_long(self.postcode_df["postcode"])
        # flood class
        risk_df = self.get_flood_class(
            self.postcode_df["postcode"], method=1).to_frame()
        # annual flood risk
        annual_risk_df = self.get_annual_flood_risk(
            self.postcode_df["postcode"]).to_frame()
        postcode_df = pd.concat([postcode_df, risk_df, annual_risk_df], axis=1)
        # convert risk label to flood probability
        risk_dict = {1: 0.00001, 2: 0.0005, 3: 0.001, 4: 0.005,
                     5: 0.01, 6: 0.015, 7: 0.02, 8: 0.03, 9: 0.04, 10: 0.05}
        postcode_df["floodProb"] = postcode_df["riskLabel"].map(
            lambda x: risk_dict[x])
        postcode_df.drop("riskLabel", axis=1)
        return postcode_df
