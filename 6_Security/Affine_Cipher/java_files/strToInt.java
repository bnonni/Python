public class strToInt {
 public String strToInt(String str) {
  char[] c = srt.toCharArray();
  for (int i = 0; i < c.length; i++) {
   int temp = c[i];
   result += temp + " ";
  }
  return result;
 }
}