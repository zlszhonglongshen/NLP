package LoadPMML;

/**
 * Created by Johnson on 18:08
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class svm_Test {
    private Evaluator modelEvaluator;

    /**
     * 通过传入 PMML 文件路径来生成机器学习模型
     *
     * @param pmmlFileName pmml 文件路径
     */
    public svm_Test(String pmmlFileName) {
        PMML pmml = null;

        try {
            if (pmmlFileName != null) {
                InputStream is = new FileInputStream(pmmlFileName);
                pmml = PMMLUtil.unmarshal(is);
                try {
                    is.close();
                } catch (IOException e) {
                    System.out.println("InputStream close error!");
                }

                ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();

                this.modelEvaluator = (Evaluator) modelEvaluatorFactory.newModelEvaluator(pmml);
                modelEvaluator.verify();
                System.out.println("加载模型成功！");
            }
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (JAXBException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    // 获取模型需要的特征名称
    public List<String> getFeatureNames() {
        List<String> featureNames = new ArrayList<String>();

        List<InputField> inputFields = modelEvaluator.getInputFields();

        for (InputField inputField : inputFields) {
            featureNames.add(inputField.getName().toString());
        }
        return featureNames;
    }

    // 获取目标字段名称
    public String getTargetName() {
        return modelEvaluator.getTargetFields().get(0).getName().toString();
    }

    // 使用模型生成预测结果
    public  Object predict(Map<FieldName, ?> arguments) {
        Map<FieldName, ?> evaluateResult = modelEvaluator.evaluate(arguments);

        FieldName fieldName = new FieldName(getTargetName());

        return  evaluateResult.get(fieldName);

    }



    public static void main(String[] args) {
//        svm_Test clf = new svm_Test("e:/xgboost.pmml");
        svm_Test clf = new svm_Test("e:/modeler.xml");

        List<String> featureNames = clf.getFeatureNames();
        System.out.println("feature: " + featureNames);

        // 构建待预测数据
        Map<FieldName, Number> waitPreSample = new HashMap <FieldName, Number>();

        waitPreSample.put(new FieldName("Clump"), 2);
        waitPreSample.put(new FieldName("UnifSize"), 1);
        waitPreSample.put(new FieldName("UnifShape"), 3);
        waitPreSample.put(new FieldName("MargAdh"), 2);
        waitPreSample.put(new FieldName("SingEpiSize"), 2);
        waitPreSample.put(new FieldName("BareNuc"), 1);
        waitPreSample.put(new FieldName("BlandChrom"), 3);
        waitPreSample.put(new FieldName("NormNucl"), 2);
        waitPreSample.put(new FieldName("Mit"), 2);

        System.out.println("waitPreSample predict result: " + clf.predict(waitPreSample).toString());

    }

}

