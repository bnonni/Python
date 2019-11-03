public class strToInt {
 public static void main(String[] args) {
  String p = "Apple";
  System.out.println(strToInt(p));
 }

 public static String strToInt(String str) {
  char[] c = str.toCharArray();
  String result = "";
  for (int i = 0; i < c.length; i++) {
   int temp = c[i];
   result += temp + " ";
  }
  return result;
 }
}