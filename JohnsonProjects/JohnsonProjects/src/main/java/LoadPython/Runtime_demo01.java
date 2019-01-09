package LoadPython;

/**
 * Created by Johnson on 19:12
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Runtime_demo01 {
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Process proc;
        int a = 2000;
        int b = 19;
        try {
            String[] args1 = new String[]{"python","e:\\add.py",String.valueOf(a),String.valueOf(b)};
            proc = Runtime.getRuntime().exec(args1);
//            proc = Runtime.getRuntime().exec("python e:\\add.py");// 执行py文件
            //用输入输出流来截取结果
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

