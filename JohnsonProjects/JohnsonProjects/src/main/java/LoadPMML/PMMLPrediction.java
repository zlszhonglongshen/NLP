package LoadPMML;

/**
 * Created by Johnson on 18:05
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.evaluator.TargetField;
import org.xml.sax.SAXException;

public class PMMLPrediction {

    public static void main(String[] args) throws Exception {
        String pathxml = "e:/xgboost.pmml";
        Map<String, Double> map = new HashMap<String, Double>();
        map.put("sepal_length", 5.1);
        map.put("sepal_width", 3.5);
        map.put("petal_length", 1.4);
        map.put("petal_width", 0.2);
        predictLrHeart(map, pathxml);
    }

    public static void predictLrHeart(Map<String, Double> irismap, String pathxml) throws Exception {

        PMML pmml;
        // 模型导入
        File file = new File(pathxml);
        InputStream is = new FileInputStream(file);
        try {
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);

            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory
                    .newInstance();
            ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory
                    .newModelEvaluator(pmml);
            Evaluator evaluator = (Evaluator) modelEvaluator;

            List<InputField> inputFields = evaluator.getInputFields();
            // 过模型的原始特征，从画像中获取数据，作为模型输入
            Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
            for (InputField inputField : inputFields) {
                FieldName inputFieldName = inputField.getName();
                Object rawValue = irismap
                        .get(inputFieldName.getValue());
                FieldValue inputFieldValue = inputField.prepare(rawValue);
                arguments.put(inputFieldName, inputFieldValue);
            }

            Map<FieldName, ?> results = evaluator.evaluate(arguments);
            List<TargetField> targetFields = evaluator.getTargetFields();
            //对于分类问题等有多个输出。
            for (TargetField targetField : targetFields) {
                FieldName targetFieldName = targetField.getName();
                Object targetFieldValue = results.get(targetFieldName);
                System.err.println("target: " + targetFieldName.getValue()
                        + " value: " + targetFieldValue);
            }
        } catch (SAXException e) {
            e.printStackTrace();
        }

    }
}