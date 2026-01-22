using OpenCvSharp;
using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        // 1) 读取图像
        string imagePath = "C:\\Users\\unimage\\Downloads\\VSCode 中 cl.exe 配置.png";
        Mat srcGray = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
        if (srcGray.Empty())
        {
            Console.WriteLine("无法读取图像，请检查路径是否正确");
            return;
        }

        // 2) 自适应二值化
        Mat binary = new Mat();
        Cv2.AdaptiveThreshold(
            srcGray, binary, 255,
            AdaptiveThresholdTypes.GaussianC,
            ThresholdTypes.BinaryInv,
            31, 7
        );

        // 3) 形态学滤波
        Mat k3 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Open, k3, iterations: 1);

        Mat k5 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Close, k5, iterations: 1);

        // 4) 轮廓检测
        Point[][] contours;
        HierarchyIndex[] hierarchy;
        Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        // 5) 结果图
        Mat result = srcGray.CvtColor(ColorConversionCodes.GRAY2BGR);

        // 6) 筛选参数
        var centers = new List<Point2f>();

        // ==================== 框选控制参数 ====================
        // 调整这些参数来控制框选区域
        int extensionWidth = 30;     // 控制框选区域的宽度（垂直于边的方向）
                                     // 值越大，框选区域越宽
        int extensionLength = 80;   // 控制框选区域的长度（沿边的方向）
                                     // 值越大，框选区域越长
        bool extendInwards = true;   // 控制框选方向
                                     // true: 向内框选（框选区域在四边形内部）
                                     // false: 向外框选（框选区域在四边形外部）
        int selectedEdgeIndex = 4;   // 选择要框选哪条边
                                     // 0: 第一条边（p1→p2）
                                     // 1: 第二条边（p2→p3）
                                     // 2: 第三条边（p3→p4）
                                     // 3: 第四条边（p4→p1）
                                     // =====================================================

        // 7) 遍历所有轮廓
        foreach (var contour in contours)
        {
            double area = Cv2.ContourArea(contour);
            if (area < 800 || area > 20000) continue;

            // 多边形近似
            double peri = Cv2.ArcLength(contour, true);
            Point[] approx = Cv2.ApproxPolyDP(contour, 0.03 * peri, true);

            if (!Cv2.IsContourConvex(approx)) continue;

            if (approx.Length == 4)
            {
                // 获取四边形的四个角点
                Point p1 = approx[0], p2 = approx[1], p3 = approx[2], p4 = approx[3];

                // 计算四边形的中心点
                Moments m = Cv2.Moments(contour);
                if (Math.Abs(m.M00) < 1e-6) continue;
                Point2f center = new Point2f((float)(m.M10 / m.M00), (float)(m.M01 / m.M00));
                centers.Add(center);

                // 定义四条边
                Point[][] edges = new Point[4][];
                edges[0] = new Point[] { p1, p2 };  // 边1：p1→p2
                edges[1] = new Point[] { p2, p3 };  // 边2：p2→p3
                edges[2] = new Point[] { p3, p4 };  // 边3：p3→p4
                edges[3] = new Point[] { p4, p1 };  // 边4：p4→p1

                // 确保选择的边索引在有效范围内
                if (selectedEdgeIndex < 0) selectedEdgeIndex = 0;
                if (selectedEdgeIndex > 3) selectedEdgeIndex = 3;

                // 获取选中的边
                Point startPoint = edges[selectedEdgeIndex][0];
                Point endPoint = edges[selectedEdgeIndex][1];

                // 计算边的方向向量
                Point direction = new Point(endPoint.X - startPoint.X, endPoint.Y - startPoint.Y);
                double edgeLength = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);

                if (edgeLength < 10) continue;

                // 计算单位方向向量
                double scaleFactor = 100.0;
                Point dirUnit = new Point(
                    (int)(direction.X / edgeLength * scaleFactor),
                    (int)(direction.Y / edgeLength * scaleFactor)
                );

                // 计算法向量（垂直于边的方向）
                Point normal = new Point(-direction.Y, direction.X);
                double normalLength = Math.Sqrt(normal.X * normal.X + normal.Y * normal.Y);
                Point normalUnit = new Point(
                    (int)(normal.X / normalLength * scaleFactor),
                    (int)(normal.Y / normalLength * scaleFactor)
                );

                // 确定框选方向
                int directionSign = extendInwards ? -1 : 1;

                // 计算边的中点
                Point edgeMidPoint = new Point(
                    (startPoint.X + endPoint.X) / 2,
                    (startPoint.Y + endPoint.Y) / 2
                );

                // 计算从边中点出发的向量
                Point fromMidToStart = new Point(startPoint.X - edgeMidPoint.X, startPoint.Y - edgeMidPoint.Y);
                Point fromMidToEnd = new Point(endPoint.X - edgeMidPoint.X, endPoint.Y - edgeMidPoint.Y);

                // 缩放边的端点（控制框选长度）
                float lengthScale = (float)extensionLength / 100.0f;
                Point scaledStart = new Point(
                    edgeMidPoint.X + (int)(fromMidToStart.X * lengthScale),
                    edgeMidPoint.Y + (int)(fromMidToStart.Y * lengthScale)
                );
                Point scaledEnd = new Point(
                    edgeMidPoint.X + (int)(fromMidToEnd.X * lengthScale),
                    edgeMidPoint.Y + (int)(fromMidToEnd.Y * lengthScale)
                );

                // 计算框选矩形的四个角点
                int widthOffset = extensionWidth * (int)scaleFactor / 100;
                Point[] selectionRect = new Point[4];

                selectionRect[0] = new Point(
                    scaledStart.X + normalUnit.X * directionSign * widthOffset / (int)scaleFactor,
                    scaledStart.Y + normalUnit.Y * directionSign * widthOffset / (int)scaleFactor
                );

                selectionRect[1] = new Point(
                    scaledEnd.X + normalUnit.X * directionSign * widthOffset / (int)scaleFactor,
                    scaledEnd.Y + normalUnit.Y * directionSign * widthOffset / (int)scaleFactor
                );

                selectionRect[2] = new Point(
                    scaledEnd.X - normalUnit.X * directionSign * widthOffset / (int)scaleFactor,
                    scaledEnd.Y - normalUnit.Y * directionSign * widthOffset / (int)scaleFactor
                );

                selectionRect[3] = new Point(
                    scaledStart.X - normalUnit.X * directionSign * widthOffset / (int)scaleFactor,
                    scaledStart.Y - normalUnit.Y * directionSign * widthOffset / (int)scaleFactor
                );

                // 绘制原四边形（蓝色）
                for (int i = 0; i < 4; i++)
                {
                    Cv2.Line(result, approx[i], approx[(i + 1) % 4],
                           new Scalar(255, 0, 0), 2);
                }

                // 用红色标记选中的边
                Cv2.Line(result, startPoint, endPoint, new Scalar(0, 0, 255), 3);

                // 绘制框选区域（绿色）
                for (int i = 0; i < 4; i++)
                {
                    Cv2.Line(result, selectionRect[i], selectionRect[(i + 1) % 4],
                           new Scalar(0, 255, 0), 2);
                }

                // 填充框选区域（半透明）
                Mat overlay = result.Clone();
                Cv2.FillPoly(overlay, new Point[][] { selectionRect },
                           new Scalar(0, 255, 0, 100));
                Cv2.AddWeighted(overlay, 0.3, result, 0.7, 0, result);

                // 在选中的边端点画标记
                Cv2.Circle(result, startPoint, 8, new Scalar(255, 255, 0), -1);
                Cv2.Circle(result, endPoint, 8, new Scalar(255, 255, 0), -1);

                // 在每条边上添加标签
                for (int i = 0; i < 4; i++)
                {
                    Point edgeMid = new Point(
                        (edges[i][0].X + edges[i][1].X) / 2,
                        (edges[i][0].Y + edges[i][1].Y) / 2
                    );
                    string label = $"E{i}";
                    Scalar color = (i == selectedEdgeIndex) ?
                        new Scalar(0, 0, 255) : new Scalar(200, 200, 200);
                    Cv2.PutText(result, label, edgeMid,
                              HersheyFonts.HersheySimplex, 0.6, color, 2);
                }

                // 显示中心点
                Cv2.Circle(result, (Point)center, 6, Scalar.Red, -1);
                Cv2.PutText(result, $"({center.X:F0},{center.Y:F0})",
                    new Point((int)center.X + 8, (int)center.Y - 8),
                    HersheyFonts.HersheySimplex, 0.6, Scalar.Red, 2);
            }
        }

        // 8) 打印中心点和参数信息
        Console.WriteLine($"检测到候选区域数量: {centers.Count}");
        for (int i = 0; i < centers.Count; i++)
        {
            Console.WriteLine($"[{i}] center=({centers[i].X:F2}, {centers[i].Y:F2})");
        }

        // 9) 保存参数信息到图像
        string paramText = $"Width: {extensionWidth}, Length: {extensionLength}, Edge: E{selectedEdgeIndex}, Inward: {extendInwards}";
        Cv2.PutText(result, paramText, new Point(20, 40), HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 255, 255), 2);

        // 10) 保存结果
        string outPath = "result_with_controlled_selection.png";
        Cv2.ImWrite(outPath, result);
        Console.WriteLine($"\n已保存: {outPath}");

        // 11) 显示结果
        Cv2.NamedWindow("检测结果", WindowFlags.Normal);
        Cv2.ImShow("检测结果", result);
        Cv2.ResizeWindow("检测结果", 1200, 900);

        Cv2.WaitKey(0);
        Cv2.DestroyAllWindows();
    }
}