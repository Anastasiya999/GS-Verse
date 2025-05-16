using System;

namespace GaussianSplatting.Runtime.Utils
{
    public enum DataQuality
    {
        VeryHigh,
        High,
        Medium,
        Low,
        VeryLow,
        Custom,
    }

    public class Quality
    {

        public static (GaussianSplatAsset.VectorFormat FormatPos, GaussianSplatAsset.VectorFormat FormatScale, GaussianSplatAsset.ColorFormat FormatColor, GaussianSplatAsset.SHFormat FormatSH) GetFormatFromQualityLevel(DataQuality quality)
        {
            switch (quality)
            {
                case DataQuality.Custom:
                    return (default, default, default, default);
                case DataQuality.VeryLow: // 18.62x smaller, 32.27 PSNR
                    return (GaussianSplatAsset.VectorFormat.Norm11, GaussianSplatAsset.VectorFormat.Norm6, GaussianSplatAsset.ColorFormat.BC7, GaussianSplatAsset.SHFormat.Cluster4k);
                case DataQuality.Low: // 14.01x smaller, 35.17 PSNR
                    return (GaussianSplatAsset.VectorFormat.Norm11, GaussianSplatAsset.VectorFormat.Norm6, GaussianSplatAsset.ColorFormat.Norm8x4, GaussianSplatAsset.SHFormat.Cluster16k);
                case DataQuality.Medium: // 5.14x smaller, 47.46 PSNR
                    return (GaussianSplatAsset.VectorFormat.Norm11, GaussianSplatAsset.VectorFormat.Norm11, GaussianSplatAsset.ColorFormat.Norm8x4, GaussianSplatAsset.SHFormat.Norm6);
                case DataQuality.High: // 2.94x smaller, 57.77 PSNR
                    return (GaussianSplatAsset.VectorFormat.Norm16, GaussianSplatAsset.VectorFormat.Norm16, GaussianSplatAsset.ColorFormat.Float16x4, GaussianSplatAsset.SHFormat.Norm11);
                case DataQuality.VeryHigh: // 1.05x smaller
                    return (GaussianSplatAsset.VectorFormat.Float32, GaussianSplatAsset.VectorFormat.Float32, GaussianSplatAsset.ColorFormat.Float32x4, GaussianSplatAsset.SHFormat.Float32);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}