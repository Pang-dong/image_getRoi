using OpenCvSharp;
using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        // 1) 读取图像（灰度用于检测；彩色用于可视化也可以）
        string imagePath = "C:\\Users\\unimage\\Pictures\\111.bmp";  // 图片路径
        Mat srcGray = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
        if (srcGray.Empty())
        {
            Console.WriteLine("无法读取图像，请检查路径是否正确");
            return;
        }

        // 2) 自适应二值化（BinaryInv：黑块变白，背景变黑）
        Mat binary = new Mat();
        Cv2.AdaptiveThreshold(
            srcGray, binary, 255,
            AdaptiveThresholdTypes.GaussianC,
            ThresholdTypes.BinaryInv,
            31, 5
        );

        // 3) 形态学滤波：开运算去小噪声 + 轻微闭运算补断裂（可选）
        Mat k3 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Open, k3, iterations: 1);

        Mat k5 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Close, k5, iterations: 1);

        // 4) 轮廓检测（外轮廓即可）
        Point[][] contours;
        HierarchyIndex[] hierarchy;
        Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        // 5) 结果图（全图分辨率，不裁剪）
        Mat result = srcGray.CvtColor(ColorConversionCodes.GRAY2BGR);

        // 6) 筛选 + 中心点
        var centers = new List<Point2f>();

        // 你可以根据实际图卡尺寸调这些阈值
        const double minArea = 400;      // 过滤小噪声
        const double maxArea = 20000;    // 过滤过大区域
        const double minAspect = 0.50;   // 允许透视畸变更大一点
        const double maxAspect = 1.70;

        foreach (var contour in contours)
        {
            double area = Cv2.ContourArea(contour);
            if (area < minArea || area > maxArea) continue;

            // 6.1) 多边形近似：找“四边形”或接近四边形
            double peri = Cv2.ArcLength(contour, true);
            Point[] approx = Cv2.ApproxPolyDP(contour, 0.03 * peri, true);

            // 只考虑凸形（排除奇怪噪声）
            if (!Cv2.IsContourConvex(approx)) continue;

            // 允许 4~6 个点（透视/锯齿可能导致不是严格4点）
            if (approx.Length < 4 || approx.Length > 6) continue;

            // 6.2) 旋转矩形（对斜边更稳）
            RotatedRect rr = Cv2.MinAreaRect(contour);
            double w = rr.Size.Width;
            double h = rr.Size.Height;
            if (w < 1 || h < 1) continue;

            double aspect = w > h ? (w / h) : (h / w);
            if (aspect < 1.0) aspect = 1.0 / aspect; // 保险
            if (aspect < 1.0) aspect = 1.0;          // 保险

            // 这里用“接近正方形/菱形”的宽高比范围
            if (aspect < minAspect || aspect > maxAspect) continue;

            // 6.3) 填充度/矩形度（抑制噪声）
            // extent = area / (boundingRect面积)；太小说明很细碎
            Rect br = Cv2.BoundingRect(contour);
            double extent = area / (br.Width * br.Height + 1e-6);
            if (extent < 0.35) continue;

            // solidity = area / convexHullArea；太小说明轮廓很凹/噪
            Point[] hull = Cv2.ConvexHull(contour);
            double hullArea = Cv2.ContourArea(hull);
            double solidity = area / (hullArea + 1e-6);
            if (solidity < 0.85) continue;

            // 6.4) 中心点（Moments）
            Moments m = Cv2.Moments(contour);
            if (Math.Abs(m.M00) < 1e-6) continue;

            Point2f center = new Point2f((float)(m.M10 / m.M00), (float)(m.M01 / m.M00));
            centers.Add(center);

            // 可视化：只画中心点（如果你不想画矩形边框就不要画）
            Cv2.Circle(result, (Point)center, 6, Scalar.Red, -1);
            Cv2.PutText(
                result, $"({center.X:F0},{center.Y:F0})",
                new Point((int)center.X + 8, (int)center.Y - 8),
                HersheyFonts.HersheySimplex, 0.6, Scalar.Red, 2
            );
        }

        // 7) 打印中心点
        Console.WriteLine($"检测到候选区域数量: {centers.Count}");
        for (int i = 0; i < centers.Count; i++)
        {
            Console.WriteLine($"[{i}] center=({centers[i].X:F2}, {centers[i].Y:F2})");
        }

        // 8) 保存全分辨率结果（不受 ImShow 影响）
        string outPath = "result_with_centers.png";
        Cv2.ImWrite(outPath, result);
        Console.WriteLine($"已保存: {outPath}");

        // 9) 关键：用可缩放窗口显示“全图”
        // 你看到“四分之一”的根因通常就在这里
        Cv2.NamedWindow("检测结果", WindowFlags.Normal);   // 可缩放
        Cv2.ImShow("检测结果", result);

        // 如果窗口太大，看不全：强制把窗口调小（不改变图像本身）
        Cv2.ResizeWindow("检测结果", 1200, 900);

        Cv2.WaitKey(0);
        Cv2.DestroyAllWindows();
    }
}
