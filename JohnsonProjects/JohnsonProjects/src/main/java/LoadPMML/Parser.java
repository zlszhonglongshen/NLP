package LoadPMML;
import java.io.*;
import java.util.HashMap;
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
import org.jpmml.model.PMMLUtil;
import com.alibaba.fastjson.JSON;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;


/**
 * Created by Johnson on 15:36
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */
public class Parser {
    private Map<FieldName,Object> paramsMap;
    private Map<String,Object> jsonMap;
    public Evaluator modelEvaluator;

    public Parser(String pmmlFileName){
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
                paramsMap = new HashMap<FieldName, Object>(2000);
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
    private void prepare(String json){
        //clear paramsMap
        paramsMap.clear();

    }
    
}
