package LoadPython;

/**
 * Created by Johnson on 18:51
 * PROJECT_NAME JohnsonProjects
 * Email:593956670@qq.com
 */
import javax.script.*;

import org.python.core.Py;
import org.python.core.PyFunction;
import org.python.core.PyInteger;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;

import java.io.*;
import static java.lang.System.*;
public class add {
    public static void main(String args[]){
        PythonInterpreter interpreter = new PythonInterpreter();
        interpreter.execfile("e:/add.py");
        PyFunction func = (PyFunction)interpreter.get("add",PyFunction.class);
        int a = 2000,b = 19;
        PyObject pyobj = func.__call__(new PyInteger(a),new PyInteger(b));
        System.out.println("answer = "+pyobj.toString());
    }
}
