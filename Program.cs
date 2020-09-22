using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace searchdata
{
    class FeatureInput
    {
        [LoadColumn(0)]
        public string FeatureName { get; set; }

        [LoadColumn(1)]
        public string ProductName { get; set; }
    }

    public class TextData
    {
        public string Text { get; set; }
    }

    public class TextTokens
    {
        public string[] Tokens { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromTextFile<FeatureInput>("./featuredata.csv", hasHeader: true, separatorChar: ',');

            //Get top 10 record
            /*
            var rows = context.Data.TakeRows(data, 10);
            var preview = context.Data.CreateEnumerable<FeatureInput>(rows, reuseRowObject: false);*/

            //Show grouping based on Product                      
            Console.WriteLine(" <== Display record based on particular Product ==> ");
            var row = DataFilterGroupByProduct(context, data, "Product A");
            var preview = context.Data.CreateEnumerable<FeatureInput>(row, reuseRowObject: false);

            foreach (var item in preview)
            {
                Console.Write($"{item.FeatureName} - {item.ProductName}");
                Console.WriteLine("");
            }

            Console.WriteLine("\n\n");           
            

            string inputText = "DeploymentActivities";
            var grpRows = DataFilterText(context, data, inputText);
            var previewGP = context.Data.CreateEnumerable<FeatureInput>(grpRows, reuseRowObject: false);

            Console.WriteLine(" <== Display record based on feature categorization ==> ");
            List<FeatureInput> objFICollection = new List<FeatureInput>();
            foreach (var item in previewGP)
            {
                Console.Write($"{item.FeatureName} - {item.ProductName}");
                Console.WriteLine("");
            }
        }

        public static IDataView DataFilter(MLContext context, IDataView data, string inputText)
        {
            var rows1 = context.Data.FilterByCustomPredicate<FeatureInput>(data, input =>
                        {
                            return input.FeatureName == inputText;
                        });

            return rows1;
        }

        public static IDataView DataFilterGroupByProduct(MLContext context, IDataView data, string inputText)
        {
            var rows1 = context.Data.FilterByCustomPredicate<FeatureInput>(data, input =>
                        {
                            return input.ProductName != inputText;
                        });

            return rows1;
        }

        public static IDataView DataFilterText(MLContext context, IDataView data, string inputText)
        {
            var rows1 = context.Data.FilterByCustomPredicate<FeatureInput>(data, input =>
                        {
                            //To extract word from line
                            var emptyData = new List<TextData>();
                            var data = context.Data.LoadFromEnumerable(emptyData);
                            var tokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' });
                            var tokenModel = tokenization.Fit(data);
                            var engine = context.Model.CreatePredictionEngine<TextData, TextTokens>(tokenModel);
                            var tokens = engine.Predict(new TextData { Text = input.FeatureName });
                            
                            string strtype = "";
                            switch (inputText)
                            {
                                case "Notification":
                                    {
                                        strtype = "email,sms,call";
                                        break;
                                    }
                                case "Financials":
                                    {
                                        strtype = "payment,quarterly,return";
                                        break;
                                    }
                                case "DeploymentActivities":
                                    {
                                        strtype = "CI/CD,Docker,Container";
                                        break;
                                    }
                                case "HumanCapitalManagement":
                                    {
                                        strtype = "Workday,employees";
                                        break;
                                    }
                            }

                            bool isSuccess = false;
                            string[] strTypes = strtype.Split(',');
                            foreach (var item in tokens.Tokens)
                            {
                                foreach (var v in strTypes)
                                {
                                    if (v.ToLower() == item.ToLower())
                                    {
                                        isSuccess = true;
                                        break;
                                    }
                                }
                                if (isSuccess == true)
                                    break;
                            }

                            if (isSuccess)
                                return input.ProductName == inputText;
                            else
                                return input.ProductName != inputText;

                        });

            return rows1;
        }
    }
}
