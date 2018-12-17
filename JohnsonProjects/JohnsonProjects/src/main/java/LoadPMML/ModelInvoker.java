package LoadPMML;

/**
 * Created by Johnson on 12:45
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.io.FileInputStream;
import javax.xml.bind.JAXBException;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;


public class ModelInvoker {
    private Evaluator modelEvaluator;

    // 通过文件读取模型
    public ModelInvoker(String pmmlFileName) {
        PMML pmml = null;
        InputStream IS = null;
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

    // 通过输入流读取模型
    public ModelInvoker(InputStream IS) {
        PMML pmml = null;
        try {
            pmml = PMMLUtil.unmarshal(IS);
            try {
                IS.close();
            } catch (IOException localIOException) {
            }
            this.modelEvaluator = ModelEvaluatorFactory.newInstance().newModelEvaluator(pmml);
        } catch (SAXException e) {
            pmml = null;
        } catch (JAXBException e) {
            pmml = null;
        } finally {
            try {
                IS.close();
            } catch (IOException localIOException3) {
            }
        }
        this.modelEvaluator.verify();
    }

    public Map<FieldName, ?> invoke(Map<FieldName, Object> paramsMap) {
        return this.modelEvaluator.evaluate(paramsMap);
    }
}
