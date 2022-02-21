
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace DocumentImageProcessing
{
    public partial class Form1 : Form
    {
        Image<Bgr, Byte>? img = null;
        Image<Gray, Byte>? grayscaled = null;
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var vid = new VideoCapture();
            do
            {
                Thread.Sleep(10);
                GetImageConturs(vid);

            } while (true);
        }

        private void GetImageConturs(VideoCapture capture)
        {
            using (var nextFrame = capture.QueryFrame())
            {
                if (nextFrame == null) return;
                using (Bitmap bitmap = nextFrame.ToBitmap())
                using (var image = bitmap.ToImage<Bgr, byte>())
                using (var grayScaleImage = image.Convert<Gray, byte>())
                using (var blurredImage = grayScaleImage.SmoothGaussian(5, 5, 0, 0))
                using (var cannyImage = new UMat())
                {
                    CvInvoke.Canny(blurredImage, cannyImage, 20, 150);
                    Emgu.CV.Util.VectorOfVectorOfPoint contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
                    CvInvoke.FindContours(cannyImage, contours, null, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
                    var cont = GetLargestContur(contours);
                    if (cont == null)
                    {
                        pictureBox1.Image = image.ToBitmap();
                        Application.DoEvents();
                        return;
                    }
                    var cntmax = new VectorOfVectorOfPoint(cont);



                    CvInvoke.DrawContours(image, cntmax, -1, new MCvScalar(255, 0, 0), 5);
                    // var rect = CvInvoke.BoundingRectangle(cont);
                    CvInvoke.Imshow("gray", grayScaleImage);
                    CvInvoke.Imshow("blurredImage", grayScaleImage);
                    CvInvoke.Imshow("canny", cannyImage);
                    pictureBox1.Image = image.ToBitmap();
                    Application.DoEvents();

                }
            }
           
        }

        private static Image cropImage(Image img, Rectangle cropArea)
        {
            Bitmap bmpImage = new Bitmap(img);
            return bmpImage.Clone(cropArea, bmpImage.PixelFormat);
        }

        private VectorOfPoint? GetLargestContur(VectorOfVectorOfPoint contours)
        {
            List<Tuple<VectorOfPoint, double>> largestcontoure = new List<Tuple<VectorOfPoint, double>>();
            for (int i = 0; i < contours.Size; i++)
            {
                using (var contour = new VectorOfPoint())
                {
                    double area = CvInvoke.ContourArea(contours[i]);
                    var peri = CvInvoke.ArcLength(contours[i], true);
                    CvInvoke.ApproxPolyDP(contours[i], contour, 0.1 * peri, true);
                    if (contour != null && contour.ToArray().Length == 4 && CvInvoke.IsContourConvex(contour))
                        largestcontoure.Add(new Tuple<VectorOfPoint, double>(contours[i], area));
                }
            }
            return largestcontoure.Where(x => x.Item2 == largestcontoure.Max(x => x.Item2)).Select(x => x.Item1).FirstOrDefault();
        }

        private void Form1_Load(object sender, EventArgs e)
        {


        }
    }
}