public class TrimIt {
 public static void main(String[] args) {
  String value = " trim this bitch   ";
  int len = value.length();
  int st = 0;
  char[] val = value;

  while ((st < len) && (val[st] <= ' ')) {
   st++;
  }
  while ((st < len) && (val[len - 1] <= ' ')) {
   len--;
  }
  System.out.println(((st > 0) || (len < value.length)) ? value.substring(st, len) : value);
 }
}