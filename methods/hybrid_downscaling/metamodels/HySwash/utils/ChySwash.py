import numpy as np
import os
import xarray as xr
from bluemath_tk.wrappers.swash.swash_wrapper import SwashModelWrapper
class ChySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with friction.
    """

    default_Cf = 0.0002

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        super().build_case(case_context=case_context, case_dir=case_dir)

        # Build the input friction file
        friction = np.ones((len(self.depth_array))) * self.default_Cf
        friction[
            int(self.fixed_parameters["Cf_ini"]) : int(self.fixed_parameters["Cf_fin"])
        ] = case_context["Cf"]
        np.savetxt(os.path.join(case_dir, "friction.txt"), friction, fmt="%.6f")  


def rbf_pca_predict(df_dataset,pca,rbf):
    # Apply PCA to the dataset
    predicted_Setup_coef=rbf.predict(dataset=df_dataset)
    predicted_Setup_ds = xr.Dataset(
        {
            "PCs": (["case_num", "n_component"], predicted_Setup_coef.values),
        },
        coords={
            "case_num": np.arange(predicted_Setup_coef.values.shape[0]),
            "n_component": np.arange(predicted_Setup_coef.values.shape[1]),
        },
    )
    ds_output = pca.inverse_transform(PCs=predicted_Setup_ds)
    return ds_output
        